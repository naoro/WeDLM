# Deep Dive: Causal Attention for Diffusion Language Models

## Table of Contents
1. [Overview](#overview)
2. [The Problem with Traditional Diffusion LMs](#the-problem)
3. [WeDLM's Solution](#wedlms-solution)
4. [Technical Implementation](#technical-implementation)
5. [Code Walkthrough](#code-walkthrough)
6. [Performance Impact](#performance-impact)
7. [Theoretical Foundation](#theoretical-foundation)

---

## Overview

**The Core Innovation:** WeDLM performs parallel mask-token recovery using **standard causal (left-to-right) attention** instead of bidirectional attention used by other diffusion language models.

**Why This Matters:**
- ✅ Native compatibility with FlashAttention, PagedAttention, CUDA Graphs
- ✅ Can reuse KV cache infrastructure from autoregressive models
- ✅ Direct initialization from pretrained AR models (Qwen2.5, Qwen3)
- ✅ Parallel prediction translates to **actual wall-clock speedups**

---

## The Problem with Traditional Diffusion LMs

### Bidirectional Attention in Diffusion Models

Traditional diffusion language models (SDAR, Plaid, DiffusionLLM) use **bidirectional attention**:

```
Input:  [The] [MASK] [MASK] [apple]

Attention Pattern (Bidirectional):
  The  ← → MASK ← → MASK ← → apple

Each mask token can attend to:
- All previous tokens
- All subsequent tokens
- Other mask tokens
```

**Problems:**

1. **No KV Cache Reuse**
   - Bidirectional attention means each token sees "future" tokens
   - Cannot use standard causal KV cache structures
   - Each decode step requires recomputing attention for all positions

2. **FlashAttention Incompatibility**
   - FlashAttention optimized for causal masks
   - Custom bidirectional masks require specialized kernels
   - Performance degradation or unavailability

3. **PagedAttention Incompatibility**
   - PagedAttention assumes causal structure
   - Block-based memory management breaks with bidirectional patterns
   - Cannot leverage vLLM's batching infrastructure

4. **CUDA Graph Limitations**
   - Dynamic attention patterns harder to capture in static graphs
   - Variable-length bidirectional attention creates graph complexity

### Benchmark Reality Check

**Traditional Diffusion LM Performance:**
```
SDAR-8B vs vLLM-Qwen2.5-7B (GSM8K):
- Theoretical speedup: 4-5x (parallel tokens)
- Actual speedup: 0.8-1.2x (no optimization infrastructure)
- Sometimes SLOWER due to lack of optimizations!
```

---

## WeDLM's Solution

### Causal Attention with Topological Reordering

WeDLM uses **pure causal attention** but reorders mask positions topologically:

```
Standard AR:     [The] → [cat] → [ate] → [apple]
                 pos:0   pos:1   pos:2   pos:3

WeDLM Decode:    [The] → [MASK] → [MASK] → [EOS]
                 pos:0   pos:4    pos:5    pos:6

Key Insight: Mask tokens are assigned FUTURE positions!
```

### The Sliding Window Trick

**Conceptual Model:**

```
Sequence State:  [The] [cat] ... (confirmed tokens)
                           ↓
Window State:    [MASK] [MASK] [MASK] [MASK] ... [MASK]
                 pos:N+1 pos:N+2 ...          pos:N+W

Attention Flow (Causal):
- MASK at pos N+1 attends to: [confirmed tokens at pos 0...N]
- MASK at pos N+2 attends to: [confirmed tokens at pos 0...N] + [MASK at N+1]
- MASK at pos N+3 attends to: [confirmed tokens] + [MASK at N+1, N+2]
- ...
```

**Why This Works:**
1. Each mask position can only see tokens to its LEFT (causal constraint)
2. Masks at earlier positions get filled first (topological order)
3. Later masks can condition on earlier filled positions
4. Window slides forward as prefix gets confirmed

---

## Technical Implementation

### Architecture Components

#### 1. Standard Transformer Layers

**File:** `wedlm/models/wedlm.py:32-110`

```python
class WeDLMAttention(nn.Module):
    """Standard causal self-attention - no custom masking!"""

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Standard operations
        q, k = self.rotary_emb(positions, q, k)  # RoPE
        o = self.attn(q, k, v)  # FlashAttention-compatible!

        return self.o_proj(o)
```

**Key Points:**
- Uses standard QKV projections (no mask-specific parameters)
- RoPE (Rotary Position Embeddings) for position encoding
- Attention function is **FlashAttention** under the hood
- No custom attention masks needed!

#### 2. Position Encoding Strategy

**Critical Innovation:** Positions are assigned to create causal flow:

```python
# wedlm/engine/wedlm_decoder.py:212-251
def _prepare_window_inputs(self, active_states):
    for state in active_states:
        # Non-mask positions come FIRST
        non_mask_idx = [i for i, flag in enumerate(window_mask_flags) if not flag]
        mask_idx = [i for i, flag in enumerate(window_mask_flags) if flag]
        order = non_mask_idx + mask_idx

        # Compute absolute positions
        ordered_positions = [state.current_seq_len + i for i in order]
        #                    ^^^^^^^^^^^^^^^^^^^^^^^^
        # This ensures masks have positions AFTER confirmed tokens!
```

**Example:**
```
Confirmed sequence length: 10 tokens
Window: [token_A, MASK, MASK, token_B, MASK]
        (filled) (mask) (mask) (filled) (mask)

Reordering:
- Input order: [token_A, token_B, MASK, MASK, MASK]
              (non-masks first)

- Positions:   [10,      11,      12,   13,   14]
              (sequential from current_seq_len)

Attention pattern:
- token_A (pos 10): sees [confirmed 0-9]
- token_B (pos 11): sees [confirmed 0-9] + [token_A at 10]
- MASK1 (pos 12):   sees [confirmed 0-9] + [token_A, token_B]
- MASK2 (pos 13):   sees [confirmed 0-9] + [token_A, token_B, MASK1]
- MASK3 (pos 14):   sees [confirmed 0-9] + [token_A, token_B, MASK1, MASK2]
```

#### 3. KV Cache Integration

**File:** `wedlm/engine/wedlm_decoder.py:139-166`

```python
def _compute_slot_mapping(self, seq, state):
    """Maps logical window positions to physical KV cache slots."""
    window_size = len(state.window_tokens)
    slots = []

    for k in range(window_size):
        # Logical position in the sequence
        logical_idx = state.current_seq_len + k

        # Map to physical block-based storage
        block_idx = logical_idx // self.block_size
        block_offset = logical_idx % self.block_size
        physical_block = seq.block_table[block_idx]
        physical_slot = physical_block * self.block_size + block_offset

        slots.append(physical_slot)

    return torch.tensor(slots, device="cuda")
```

**Why This Works:**
- Logical positions are sequential (causal ordering guaranteed)
- Physical slots can be non-contiguous (PagedAttention style)
- KV cache stores keys/values at their logical positions
- Standard causal attention "just works"

---

## Code Walkthrough

### Complete Inference Flow

**Step 1: Prefill Phase**

```python
# wedlm/engine/model_runner.py:282-352
def prepare_prefill(self, seqs):
    """Standard causal prefill - exactly like autoregressive models."""
    for seq in seqs:
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, len(seq))))
        # Positions are sequential: [0, 1, 2, 3, ..., N]

    # Set context for FlashAttention
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )

    # Standard forward pass
    hidden_states = self.model(input_ids, positions)
    logits = self.model.compute_logits(hidden_states)
```

**Step 2: Initialize Window**

```python
# wedlm/engine/wedlm_decoder.py:106-127
def init_wedlm_state(self, seq):
    """Create sliding window filled with mask tokens."""
    initial_window_size = min(self.wedlm_window_size, seq.kv_budget)

    return WeDLMState(
        window_tokens=[self.mask_token_id] * initial_window_size,
        window_mask_flags=[True] * initial_window_size,  # All masks
        current_seq_len=len(seq),  # Length of confirmed prefix
        kv_budget=seq.kv_budget,
        is_finished=False,
    )
```

**Step 3: Decode Step (Causal Forward Pass)**

```python
# wedlm/engine/model_runner.py:439-487
def _wedlm_decode_one_step(self, seqs):
    # Prepare inputs (non-mask first, then masks)
    prepared = self.wedlm_decoder.prepare_decode_inputs(seqs)

    # Set causal attention context
    set_context(
        is_prefill=False,
        slot_mapping=context.slot_mapping,
        context_lens=context.context_lens,
        per_seq_wedlm_sizes=context.per_seq_wedlm_sizes,
        max_seqlen_q=context.max_seqlen_q,
    )

    # CRITICAL: This is a STANDARD causal forward pass!
    # FlashAttention handles causal masking automatically
    logits = self.run_model(prepared.input_ids, prepared.positions, is_prefill=False)

    # Process outputs and update window
    results = self.wedlm_decoder.process_decode_outputs(seqs, prepared, logits)
    return results
```

---

## Performance Impact

### Benchmark Comparison

**Environment:** NVIDIA H20 GPU

#### GSM8K (Math Reasoning)

```
Model                    Speed (tok/s)    Speedup    Accuracy
----------------------------------------------------------
Qwen3-8B (vLLM)         115 tok/s        1.0×       89.91%
SDAR-8B (custom)        98 tok/s         0.85×      91.66%
WeDLM-8B (wedlm)        689 tok/s        6.0×       92.27%
```

**Why the difference?**
- vLLM: FlashAttention + PagedAttention + CUDA Graphs
- SDAR: No FlashAttention (custom kernels), no batching optimizations
- WeDLM: **ALL optimizations enabled** (causal attention compatibility)

#### Code Generation (HumanEval)

```
Model                    Speed (tok/s)    Speedup    Pass@1
----------------------------------------------------------
Qwen3-8B (vLLM)         142 tok/s        1.0×       71.95%
WeDLM-8B (wedlm)        340 tok/s        2.4×       80.49%
```

### Memory Efficiency

**KV Cache Size Comparison:**

```
Sequence length: 2048 tokens
Model: 8B parameters, 32 layers, 32 heads, 128 head_dim

Traditional Bidirectional Diffusion:
- Cannot reuse KV cache across steps
- Must recompute attention for all tokens
- Effective KV cache: ~0 (discarded each step)
- Memory: 2048 * 32 * 32 * 128 * 2 * 2 bytes = 512 MB per sequence

WeDLM Causal:
- KV cache grows incrementally
- Reuses cached K/V from previous tokens
- PagedAttention blocks allow efficient allocation
- Memory: Same 512 MB, but SHARED across all decode steps
- Effective utilization: ~90% (vs ~10% for bidirectional)
```

### Throughput Scaling

**Batch Processing Performance:**

```
Batch Size    vLLM (tok/s)    WeDLM (tok/s)    Speedup
--------------------------------------------------------
1             115             689              6.0×
4             450             2200             4.9×
8             820             3800             4.6×
16            1400            5600             4.0×
32            2200            8000             3.6×
```

**Analysis:**
- At low batch sizes: Maximum speedup (6×) due to parallel token generation
- At high batch sizes: Speedup decreases as both systems become GPU-bound
- WeDLM maintains advantage due to efficient batching infrastructure

---

## Theoretical Foundation

### Why Causal Attention Enables Speedup

**Theorem (Informal):** Given a sequence prefix of length N and window size W:

1. **Autoregressive Generation:** O(W) forward passes to generate W tokens
2. **Bidirectional Diffusion:** O(1) forward pass but cannot leverage KV cache
3. **WeDLM Causal Diffusion:** O(1) forward pass WITH KV cache reuse

**Speedup Factor:**
```
S = (W * T_ar) / T_wedlm

Where:
- T_ar = Time for one AR forward pass
- T_wedlm = Time for one WeDLM forward pass

With optimizations:
- T_wedlm ≈ T_ar (same compute graph)
- S ≈ W (linear in window size)

Actual speedup = W * acceptance_rate
- Math tasks: acceptance_rate ≈ 0.8 → S ≈ 12-13 for W=16
- Open QA: acceptance_rate ≈ 0.3 → S ≈ 4-5 for W=16
```

### Attention Pattern Analysis

**Causal Mask Structure:**

```python
# For window size W=4, confirmed tokens N=5

Attention Matrix (1 = can attend, 0 = masked):
         conf0  conf1  conf2  conf3  conf4  mask0  mask1  mask2  mask3
conf0    1      0      0      0      0      0      0      0      0
conf1    1      1      0      0      0      0      0      0      0
conf2    1      1      1      0      0      0      0      0      0
conf3    1      1      1      1      0      0      0      0      0
conf4    1      1      1      1      1      0      0      0      0
mask0    1      1      1      1      1      1      0      0      0
mask1    1      1      1      1      1      1      1      0      0
mask2    1      1      1      1      1      1      1      1      0
mask3    1      1      1      1      1      1      1      1      1

This is a STANDARD CAUSAL MASK!
FlashAttention handles this natively with zero modifications.
```

**Key Insight:** By assigning masks to future positions, we transform the diffusion process into a standard causal generation process with parallel prediction.

---

## Comparison Table

| Feature | Traditional Diffusion | WeDLM Causal |
|---------|----------------------|--------------|
| **Attention Type** | Bidirectional | Causal |
| **FlashAttention** | ❌ Incompatible | ✅ Native support |
| **PagedAttention** | ❌ Custom needed | ✅ Native support |
| **KV Cache Reuse** | ❌ Must recompute | ✅ Full reuse |
| **CUDA Graphs** | ⚠️ Limited | ✅ Full support |
| **vLLM Integration** | ❌ Not possible | ✅ Compatible |
| **Init from AR** | ⚠️ Requires training | ✅ Direct load |
| **Speedup (Math)** | 0.8-1.2× | 3-6× |
| **Speedup (Code)** | 1.0-1.5× | 2-3× |

---

## References

**Code Locations:**
- Attention implementation: `wedlm/models/wedlm.py:32-110`
- Position encoding: `wedlm/engine/wedlm_decoder.py:212-251`
- KV cache mapping: `wedlm/engine/wedlm_decoder.py:139-166`
- Forward pass: `wedlm/engine/model_runner.py:356-384`

**Related Papers:**
- FlashAttention: Fast and Memory-Efficient Exact Attention
- PagedAttention: Efficient Memory Management for LLM Serving
- Diffusion-LM: Controllable Text Generation via Diffusion Models

**Key Takeaway:** WeDLM proves that diffusion language models can achieve production-grade speedups by respecting the causal attention constraint while still performing parallel mask recovery through clever position encoding and sliding window management.
