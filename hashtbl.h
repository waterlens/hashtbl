// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HASHTBL_H
#define HASHTBL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "komihash.h"

#if defined(__x86_64__) && defined(__SSE3__)
#include <immintrin.h>
typedef __m128i group_t;
#define GROUP_WIDTH 16
#define GROUP_SHIFT 0
#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_ACLE)
#include <arm_acle.h>
#include <arm_neon.h>
typedef uint8x8_t group_t;
#define GROUP_WIDTH 8
#define GROUP_SHIFT 3
#else
#error "Unsupported architecture"
#endif

typedef struct {
  int8_t x;
} ctrl_t;

typedef uint8_t h2_t;
#define GUARD_NUM (GROUP_WIDTH - 1)

_Static_assert(sizeof(ctrl_t) == 1, "ctrl_t must be 1 byte");

#define CTRL_EMPTY ((ctrl_t){.x = -128})
#define CTRL_DELETED ((ctrl_t){.x = -2})
#define CTRL_SENTINEL ((ctrl_t){.x = -1})

#ifdef __has_attribute
#if __has_attribute(maybe_unused)
#define MAYBEUNUSED [[maybe_unused]]
#else
#define MAYBEUNUSED __attribute__((unused))
#endif
#else
#define MAYBEUNUSED __attribute__((unused))
#endif

#define HASHTBL_META(X_, NAME_) NAME_##_##X_
#define HASHTBL_INTERNAL_META(X_, NAME_) X_##_of_##NAME_

#define HASHTBL_CHECK(cond, ...)                                               \
  do {                                                                         \
    if (cond)                                                                  \
      break;                                                                   \
    fprintf(stderr, "HASHTBL_CHECK failed at %s:%d\n", __FILE__, __LINE__);    \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
    fflush(stderr);                                                            \
    abort();                                                                   \
  } while (false)

#ifdef DEBUG
#define HASHTBL_DCHECK(cond_, ...) ((void)0)
#else
#define HASHTBL_DCHECK HASHTBL_CHECK
#endif

#define HASHTBL_ASSERT_IS_FULL(ctrl)                                           \
  HASHTBL_CHECK(                                                               \
      (ctrl) != NULL && hashtbl_ctrl_is_full(*(ctrl)),                         \
      "Invalid operation on iterator (%p/%d). The element might have "         \
      "been erased, or the table might have rehashed.",                        \
      (ctrl), (ctrl) ? (ctrl)->x : -1)

#define HASHTBL_ASSERT_IS_VALID(ctrl)                                          \
  HASHTBL_CHECK(                                                               \
      (ctrl) == NULL || hashtbl_ctrl_is_full(*(ctrl)),                         \
      "Invalid operation on iterator (%p/%d). The element might have "         \
      "been erased, or the table might have rehashed.",                        \
      (ctrl), (ctrl) ? (ctrl)->x : -1)

struct hashtbl {
  size_t size;
  size_t capacity;
  size_t available;
  uint8_t *slots;
  ctrl_t ctrl[0];
};

struct hashtbl_iterator {
  struct hashtbl *tbl;
  ctrl_t *ctrl;
  uint8_t *slot;
};

struct bitmask {
  uint64_t mask;
  uint32_t width;
  uint32_t shift;
};

struct probe_seq {
  size_t mask;
  size_t offset;
  size_t index;
};

struct find_info {
  size_t offset;
  size_t probe_length;
};

struct prepare_insert {
  size_t index;
  bool inserted;
};

struct policy {
  size_t (*hash)(const void *val);
  bool (*eq)(const void *needle, const void *candidate);
};

struct insert {
  struct hashtbl_iterator iter;
  bool inserted;
};

static inline size_t hash_seed(const ctrl_t *ctrl) {
  return ((uintptr_t)ctrl) >> 14;
}

static inline size_t h1_hash(size_t hash, const ctrl_t *ctrl) {
  return (hash >> 7) ^ hash_seed(ctrl);
}

static inline size_t h2_hash(size_t hash) { return hash & 0x7f; }

static inline ctrl_t h2_hash_as_ctrl(size_t hash) {
  return (ctrl_t){.x = (int8_t)h2_hash(hash)};
}

static inline bool hashtbl_ctrl_is_empty(ctrl_t ctrl) {
  return ctrl.x == CTRL_EMPTY.x;
}

static inline bool hashtbl_ctrl_is_deleted(ctrl_t ctrl) {
  return ctrl.x == CTRL_DELETED.x;
}

static inline bool hashtbl_ctrl_is_full(ctrl_t ctrl) { return ctrl.x >= 0; }

static inline bool hashtbl_ctrl_is_empty_or_deleted(ctrl_t ctrl) {
  return ctrl.x < CTRL_SENTINEL.x;
}

#define group_bitmask(x)                                                       \
  ((struct bitmask){(uint64_t)(x), GROUP_WIDTH, GROUP_SHIFT})

static inline uint32_t ctz64(uint64_t x) { return __builtin_ctzll(x); }

static inline uint32_t clz64(uint64_t x) {
  return x == 0 ? 64 : __builtin_clzll(x);
}

#define ctz(x) (ctz64(x))
#define clz(x)                                                                 \
  (clz64(x) - (uint32_t)((sizeof(unsigned long long) - sizeof(x)) * 8))
#define bitwidth(x) (((uint32_t)(sizeof(x) * 8)) - clz(x))

static inline uint32_t bitmask_lsb_set(const struct bitmask *bm) {
  return ctz(bm->mask) >> bm->shift;
}

static inline uint32_t bitmask_msb_set(const struct bitmask *bm) {
  return (uint32_t)(bitwidth(bm->mask) - 1) >> bm->shift;
}

static inline uint32_t bitmask_ctz(const struct bitmask *bm) {
  return ctz(bm->mask) >> bm->shift;
}

static inline uint32_t bitmask_clz(const struct bitmask *bm) {
  uint32_t total_significant_bits = bm->width << bm->shift;
  uint32_t extra_bits = sizeof(bm->mask) * 8 - total_significant_bits;
  return (uint32_t)(clz(bm->mask << extra_bits)) >> bm->shift;
}

static inline bool bitmask_next(struct bitmask *bm, uint32_t *bit) {
  if (bm->mask == 0)
    return false;

  *bit = bitmask_lsb_set(bm);
  bm->mask &= bm->mask - 1;
  return true;
}

#if defined(__x86_64__) && defined(__SSE3__)

static inline group_t hashtbl_group_new(const ctrl_t *pos) {
  return _mm_loadu_si128((const group_t *)pos);
}

static inline struct bitmask hashtbl_group_match(const group_t *gp, h2_t hash) {
  return group_bitmask(
      _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(hash), *gp)));
}

static inline struct bitmask hashtbl_group_match_empty(const group_t *gp) {
  return group_bitmask(_mm_movemask_epi8(_mm_sign_epi8(*gp, *gp)));
}

static inline struct bitmask
hashtbl_group_match_empty_or_deleted(const group_t *gp) {
  return group_bitmask(
      _mm_movemask_epi8(_mm_cmpgt_epi8(_mm_set1_epi8(CTRL_SENTINEL.x), *gp)));
}

static inline uint32_t
hashtbl_group_count_leading_empty_or_deleted(const group_t *gp) {
  return ctz((uint32_t)(_mm_movemask_epi8(_mm_cmpgt_epi8(
                            _mm_set1_epi8(CTRL_SENTINEL.x), *gp)) +
                        1));
}

static inline void
hashtbl_group_convert_special_to_empty_and_full_to_deleted(const group_t *gp,
                                                           ctrl_t *dst) {
  group_t msbs = _mm_set1_epi8((char)-128);
  group_t x126 = _mm_set1_epi8(126);
  group_t res = _mm_or_si128(_mm_shuffle_epi8(x126, *gp), msbs);
  _mm_storeu_si128((group_t *)dst, res);
}

#elif defined(__aarch64__) && defined(__ARM_NEON)

static inline group_t hashtbl_group_new(const ctrl_t *pos) {
  return vld1_u8((const uint8_t *)pos);
}

static inline struct bitmask hashtbl_group_match(const group_t *gp, h2_t hash) {
  return group_bitmask(
      vget_lane_u64(vreinterpret_u64_u8((vceq_u8(vdup_n_u8(hash), *gp))), 0) &
      0x8080808080808080ULL);
}

static inline struct bitmask hashtbl_group_match_empty(const group_t *gp) {
  return group_bitmask(
      vget_lane_u64(vreinterpret_u64_u8(vceq_s8(vdup_n_s8(CTRL_EMPTY.x),
                                                vreinterpret_s8_u8(*gp))),
                    0));
}

static inline struct bitmask
hashtbl_group_match_empty_or_deleted(const group_t *gp) {
  return group_bitmask(
      vget_lane_u64(vreinterpret_u64_u8(vcgt_s8(vdup_n_s8(CTRL_SENTINEL.x),
                                                vreinterpret_s8_u8(*gp))),
                    0));
}

static inline uint32_t
hashtbl_group_count_leading_empty_or_deleted(const group_t *gp) {
  uint64_t mask =
      vget_lane_u64(vreinterpret_u64_u8(vcle_s8(vdup_n_s8(CTRL_SENTINEL.x),
                                                vreinterpret_s8_u8(*gp))),
                    0);
  return (uint32_t)ctz(mask) >> 3;
}

static inline void
hashtbl_group_convert_special_to_empty_and_full_to_deleted(const group_t *gp,
                                                           ctrl_t *dst) {
  uint64_t mask = *(uint64_t *)gp;
  uint64_t msbs = 0x8080808080808080ULL;
  uint64_t lsbs = 0x0101010101010101ULL;
  uint64_t x = mask & msbs;
  uint64_t res = (~x + (x >> 7)) & ~lsbs;
  *(uint64_t *)dst = res;
}
#else
#error "Unsupported architecture"
#endif

#define probe_seq(x, y) ((struct probe_seq){.mask = y, .offset = x & y})

static inline size_t hashtbl_probe_seq_offset(const struct probe_seq *ps,
                                              size_t i) {
  return (ps->offset + i) & ps->mask;
}

static inline void hashtbl_probe_seq_next(struct probe_seq *ps) {
  ps->index += GROUP_WIDTH;
  ps->offset += ps->index;
  ps->offset &= ps->mask;
}

static inline struct probe_seq
hashtbl_probe_seq_start(const ctrl_t *ctrl, size_t hash, size_t capacity) {
  return probe_seq(h1_hash(hash, ctrl), capacity);
}

static inline struct find_info
hashtbl_find_first_non_full(const ctrl_t *ctrl, size_t hash, size_t capacity) {
  struct probe_seq seq = hashtbl_probe_seq_start(ctrl, hash, capacity);
  for (;;) {
    group_t g = hashtbl_group_new(ctrl + seq.offset);
    struct bitmask mask = hashtbl_group_match_empty_or_deleted(&g);
    if (mask.mask) {
      return (struct find_info){
          hashtbl_probe_seq_offset(&seq, bitmask_ctz(&mask)), seq.index};
    }
    hashtbl_probe_seq_next(&seq);
    HASHTBL_DCHECK(seq.index <= capacity, "full table!");
  }
}

static inline bool hashtbl_is_valid_capacity(size_t n) {
  return ((n + 1) & n) == 0 && n > 0;
}

static inline size_t hashtbl_random_seed(void) {
  static _Thread_local size_t counter;
  size_t value = ++counter;
  return value ^ ((size_t)&counter);
}

MAYBEUNUSED
static bool hashtbl_should_insert_backwards(size_t hash, const ctrl_t *ctrl) {
  return (h1_hash(hash, ctrl) ^ hashtbl_random_seed()) % 13 > 6;
}

MAYBEUNUSED
static void
hashtbl_convert_deleted_to_empty_and_full_to_deleted(ctrl_t *ctrl,
                                                     size_t capacity) {
  HASHTBL_DCHECK(ctrl[capacity].x == CTRL_SENTINEL.x,
                 "bad ctrl value at %zu: %02x", capacity, ctrl[capacity].x);
  HASHTBL_DCHECK(hashtbl_is_valid_capacity(capacity), "invalid capacity: %zu",
                 capacity);
  for (ctrl_t *pos = ctrl; pos < ctrl + capacity; pos += GROUP_WIDTH) {
    group_t gp = hashtbl_group_new(pos);
    hashtbl_group_convert_special_to_empty_and_full_to_deleted(&gp, pos);
  }

  memcpy(ctrl + capacity + 1, ctrl, GUARD_NUM);
  ctrl[capacity] = CTRL_SENTINEL;
}

static inline void hashtbl_reset_ctrl(size_t capacity, ctrl_t *ctrl) {
  memset(ctrl, CTRL_EMPTY.x, capacity + 1 + GUARD_NUM);
  ctrl[capacity] = CTRL_SENTINEL;
}

static inline void hashtbl_set_ctrl(size_t i, ctrl_t h, size_t capacity,
                                    ctrl_t *ctrl) {
  HASHTBL_DCHECK(i < capacity, "hashtbl_set_ctrl out-of-bounds: %zu >= %zu", i,
                 capacity);
  size_t mirrored_i = ((i - GUARD_NUM) & capacity) + (GUARD_NUM & capacity);
  ctrl[i].x = h.x;
  ctrl[mirrored_i].x = h.x;
}

static inline size_t hashtbl_normalize_capacity(size_t n) {
  return n ? SIZE_MAX >> clz(n) : 1;
}

static inline size_t hashtbl_capacity_to_growth(size_t capacity) {
  HASHTBL_DCHECK(hashtbl_is_valid_capacity(capacity), "invalid capacity: %zu",
                 capacity);
  if (GROUP_WIDTH == 8 && capacity == 7)
    return 6;
  return capacity - capacity / 8;
}

static inline size_t hashtbl_growth_to_lower_bound_capacity(size_t growth) {
  if (GROUP_WIDTH == 8 && growth == 7)
    return 8;
  return growth + (size_t)((((int64_t)growth) - 1) / 7);
}

static inline size_t hashtbl_slot_offset(size_t capacity, size_t align) {
  HASHTBL_DCHECK(hashtbl_is_valid_capacity(capacity), "invalid capacity: %zu",
                 capacity);
  size_t num = capacity + GUARD_NUM + 1;
  return (num + align - 1u) & -align;
}

#define HASHTBL_INTERNAL_ALLOC_SIZE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,      \
                                         FREE_, GET_, HASH_, INIT_, MOVE_,     \
                                         SLOT_ALIGN_, SLOT_SIZE_)              \
  static inline size_t HASHTBL_INTERNAL_META(hashtbl_alloc_size,               \
                                             NAME_)(size_t capacity) {         \
    return (sizeof(struct hashtbl) +                                           \
            hashtbl_slot_offset(capacity, SLOT_ALIGN_) +                       \
            capacity * SLOT_SIZE_);                                            \
  }

static inline bool hashtbl_is_small(size_t capacity) {
  return capacity < GROUP_WIDTH - 1;
}

#define HASHTBL_INTERNAL_ITER_SKIP_EMPTY_OR_DELETED_META(                      \
    NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,         \
    SLOT_ALIGN_, SLOT_SIZE_)                                                   \
  static inline void HASHTBL_INTERNAL_META(hashtbl_iter_skip_empty_or_deleted, \
                                           NAME_)(struct hashtbl_iterator *    \
                                                  it) {                        \
    while (hashtbl_ctrl_is_empty_or_deleted(*it->ctrl)) {                      \
      group_t g = hashtbl_group_new(it->ctrl);                                 \
      uint32_t shift = hashtbl_group_count_leading_empty_or_deleted(&g);       \
      it->ctrl += shift;                                                       \
      it->slot += shift * SLOT_SIZE_;                                          \
    }                                                                          \
                                                                               \
    if (it->ctrl->x == CTRL_SENTINEL.x) {                                      \
      it->ctrl = NULL;                                                         \
      it->slot = NULL;                                                         \
    }                                                                          \
  }

#define HASHTBL_INTERNAL_ITER_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,  \
                                      GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,  \
                                      SLOT_SIZE_)                              \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_iter_at, NAME_)(struct hashtbl * tbl, size_t index) {            \
    struct hashtbl_iterator it = {                                             \
        tbl,                                                                   \
        tbl->ctrl + index,                                                     \
        tbl->slots + index * SLOT_SIZE_,                                       \
    };                                                                         \
    HASHTBL_INTERNAL_META(hashtbl_iter_skip_empty_or_deleted, NAME_)(&it);     \
    HASHTBL_ASSERT_IS_VALID(it.ctrl);                                          \
    return it;                                                                 \
  }

#define HASHTBL_INTERNAL_ITER_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,     \
                                   GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,     \
                                   SLOT_SIZE_)                                 \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_iter, NAME_)(struct hashtbl * tbl) {                             \
    return HASHTBL_INTERNAL_META(hashtbl_iter_at, NAME_)(tbl, 0);              \
  }

#define HASHTBL_INTERNAL_CONST_ITER_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,   \
                                            FREE_, GET_, HASH_, INIT_, MOVE_,  \
                                            SLOT_ALIGN_, SLOT_SIZE_)           \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_const_iter_at, NAME_)(const struct hashtbl *tbl, size_t index) { \
    return HASHTBL_INTERNAL_META(hashtbl_iter_at,                              \
                                 NAME_)((struct hashtbl *)tbl, index);         \
  }

#define HASHTBL_INTERNAL_CONST_ITER_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,      \
                                         FREE_, GET_, HASH_, INIT_, MOVE_,     \
                                         SLOT_ALIGN_, SLOT_SIZE_)              \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_const_iter, NAME_)(const struct hashtbl *tbl) {                  \
    return HASHTBL_INTERNAL_META(hashtbl_const_iter_at, NAME_)(tbl, 0);        \
  }

#define HASHTBL_INTERNAL_ITER_GET_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, \
                                       GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_, \
                                       SLOT_SIZE_)                             \
  static inline void *HASHTBL_INTERNAL_META(hashtbl_iter_get, NAME_)(          \
      const struct hashtbl_iterator *it) {                                     \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    HASHTBL_ASSERT_IS_VALID(it->ctrl);                                         \
    if (it->slot == NULL)                                                      \
      return NULL;                                                             \
    return (void *)GET_(it->slot);                                             \
  }

#define HASHTBL_INTERNAL_ITER_NEXT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,       \
                                        FREE_, GET_, HASH_, INIT_, MOVE_,      \
                                        SLOT_ALIGN_, SLOT_SIZE_)               \
  static inline void *HASHTBL_INTERNAL_META(hashtbl_iter_next, NAME_)(         \
      struct hashtbl_iterator * it) {                                          \
    HASHTBL_ASSERT_IS_FULL(it->ctrl);                                          \
    it->ctrl += 1;                                                             \
    it->slot += SLOT_SIZE_;                                                    \
                                                                               \
    HASHTBL_INTERNAL_META(hashtbl_iter_skip_empty_or_deleted, NAME_)(it);      \
    return HASHTBL_INTERNAL_META(hashtbl_iter_get, NAME_)(it);                 \
  }

static inline void hashtbl_erase_meta_only(struct hashtbl_iterator it) {
  HASHTBL_DCHECK(hashtbl_ctrl_is_full(*it.ctrl), "erasing a dangling iterator");
  --it.tbl->size;

  size_t index = (size_t)(it.ctrl - it.tbl->ctrl);
  size_t index_before = (index - GROUP_WIDTH) & it.tbl->capacity;

  group_t gp_after = hashtbl_group_new(it.ctrl);
  struct bitmask empty_after = hashtbl_group_match_empty(&gp_after);

  group_t gp_before = hashtbl_group_new(it.tbl->ctrl + index_before);
  struct bitmask empty_before = hashtbl_group_match_empty(&gp_before);

  bool was_never_full = empty_before.mask && empty_after.mask &&
                        (size_t)(bitmask_ctz(&empty_after) +
                                 bitmask_clz(&empty_before)) < GROUP_WIDTH;

  hashtbl_set_ctrl(index, was_never_full ? CTRL_EMPTY : CTRL_DELETED,
                   it.tbl->capacity, it.tbl->ctrl);
  it.tbl->available += was_never_full;
}

static inline void hashtbl_reset_available(struct hashtbl *tbl) {
  tbl->available = hashtbl_capacity_to_growth(tbl->capacity) - tbl->size;
}

#define HASHTBL_INTERNAL_ALLOC_NEW_TABLE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, \
                                              FREE_, GET_, HASH_, INIT_,       \
                                              MOVE_, SLOT_ALIGN_, SLOT_SIZE_)  \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(                         \
      hashtbl_alloc_new_table, NAME_)(size_t capacity, size_t size) {          \
    HASHTBL_DCHECK(capacity, "capacity should be nonzero");                    \
    struct hashtbl *tbl = (struct hashtbl *)ALLOC_(                            \
        HASHTBL_INTERNAL_META(hashtbl_alloc_size, NAME_)(capacity));           \
                                                                               \
    tbl->size = size;                                                          \
    tbl->capacity = capacity;                                                  \
    tbl->slots = (uint8_t *)tbl->ctrl +                                        \
                 hashtbl_slot_offset(tbl->capacity, SLOT_ALIGN_);              \
    hashtbl_reset_ctrl(tbl->capacity, tbl->ctrl);                              \
    hashtbl_reset_available(tbl);                                              \
    return tbl;                                                                \
  }

#define HASHTBL_INTERNAL_DESTROY_SLOTS_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,   \
                                            FREE_, GET_, HASH_, INIT_, MOVE_,  \
                                            SLOT_ALIGN_, SLOT_SIZE_)           \
  static inline void HASHTBL_INTERNAL_META(hashtbl_destroy_slots,              \
                                           NAME_)(struct hashtbl * tbl) {      \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    if (!tbl->capacity)                                                        \
      return;                                                                  \
                                                                               \
    for (size_t i = 0; i != tbl->capacity; ++i) {                              \
      if (hashtbl_ctrl_is_full(tbl->ctrl[i]))                                  \
        DEL_(tbl->slots + i * SLOT_SIZE_);                                     \
    }                                                                          \
  }

#define HASHTBL_INTERNAL_RESIZE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,   \
                                     GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,   \
                                     SLOT_SIZE_)                               \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(hashtbl_resize, NAME_)(  \
      struct hashtbl * tbl, size_t new_capacity) {                             \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    HASHTBL_DCHECK(hashtbl_is_valid_capacity(new_capacity),                    \
                   "invalid capacity: %zu", new_capacity);                     \
    ctrl_t *old_ctrl = tbl->ctrl;                                              \
    uint8_t *old_slots = tbl->slots;                                           \
    size_t old_capacity = tbl->capacity;                                       \
                                                                               \
    struct hashtbl *new_tbl = HASHTBL_INTERNAL_META(                           \
        hashtbl_alloc_new_table, NAME_)(new_capacity, tbl->size);              \
                                                                               \
    MAYBEUNUSED size_t total_probe_length = 0;                                 \
    for (size_t i = 0; i != old_capacity; ++i) {                               \
      if (hashtbl_ctrl_is_full(old_ctrl[i])) {                                 \
        size_t hash = HASH_(GET_(old_slots + i * SLOT_SIZE_));                 \
        struct find_info target = hashtbl_find_first_non_full(                 \
            new_tbl->ctrl, hash, new_tbl->capacity);                           \
        size_t new_i = target.offset;                                          \
        total_probe_length += target.probe_length;                             \
        hashtbl_set_ctrl(new_i, h2_hash_as_ctrl(hash), new_tbl->capacity,      \
                         new_tbl->ctrl);                                       \
        MOVE_(new_tbl->slots + new_i * SLOT_SIZE_,                             \
              old_slots + i * SLOT_SIZE_);                                     \
      }                                                                        \
    }                                                                          \
    FREE_(tbl,                                                                 \
          HASHTBL_INTERNAL_META(hashtbl_alloc_size, NAME_)(old_capacity));     \
    return new_tbl;                                                            \
  }

#define HASHTBL_INTERNAL_DROP_DELETES_WITHOUT_RESIZE_META(                     \
    NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,         \
    SLOT_ALIGN_, SLOT_SIZE_)                                                   \
  static void HASHTBL_INTERNAL_META(hashtbl_drop_deletes_without_resize,       \
                                    NAME_)(struct hashtbl * tbl) {             \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    HASHTBL_DCHECK(hashtbl_is_valid_capacity(tbl->capacity),                   \
                   "invalid capacity: %zu", tbl->capacity);                    \
    HASHTBL_DCHECK(!hashtbl_is_small(tbl->capacity),                           \
                   "unexpected small capacity: %zu", tbl->capacity);           \
    hashtbl_convert_deleted_to_empty_and_full_to_deleted(tbl->ctrl,            \
                                                         tbl->capacity);       \
                                                                               \
    MAYBEUNUSED size_t total_probe_length = 0;                                 \
    void *slot = ALLOC_(SLOT_SIZE_);                                           \
                                                                               \
    for (size_t i = 0; i != tbl->capacity; ++i) {                              \
      if (!hashtbl_ctrl_is_deleted(tbl->ctrl[i]))                              \
        continue;                                                              \
                                                                               \
      uint8_t *old_slot = tbl->slots + i * SLOT_SIZE_;                         \
      size_t hash = HASH_(GET_(old_slot));                                     \
                                                                               \
      const struct find_info target =                                          \
          hashtbl_find_first_non_full(tbl->ctrl, hash, tbl->capacity);         \
      const size_t new_i = target.offset;                                      \
      total_probe_length += target.probe_length;                               \
                                                                               \
      uint8_t *new_slot = tbl->slots + new_i * SLOT_SIZE_;                     \
                                                                               \
      const size_t probe_offset =                                              \
          hashtbl_probe_seq_start(tbl->ctrl, hash, tbl->capacity).offset;      \
                                                                               \
      if ((((new_i - probe_offset) & tbl->capacity) / GROUP_WIDTH) ==          \
          (((i - probe_offset) & tbl->capacity) / GROUP_WIDTH)) {              \
        hashtbl_set_ctrl(i, h2_hash_as_ctrl(hash), tbl->capacity, tbl->ctrl);  \
        continue;                                                              \
      }                                                                        \
                                                                               \
      if (hashtbl_ctrl_is_empty(tbl->ctrl[new_i])) {                           \
        hashtbl_set_ctrl(new_i, h2_hash_as_ctrl(hash), tbl->capacity,          \
                         tbl->ctrl);                                           \
        MOVE_(new_slot, old_slot);                                             \
        hashtbl_set_ctrl(i, CTRL_EMPTY, tbl->capacity, tbl->ctrl);             \
      } else {                                                                 \
        HASHTBL_DCHECK(hashtbl_ctrl_is_deleted(tbl->ctrl[new_i]),              \
                       "bad ctrl value at %zu: %02x", new_i,                   \
                       tbl->ctrl[new_i].x);                                    \
        hashtbl_set_ctrl(new_i, h2_hash_as_ctrl(hash), tbl->capacity,          \
                         tbl->ctrl);                                           \
        MOVE_(slot, old_slot);                                                 \
        MOVE_(old_slot, new_slot);                                             \
        MOVE_(new_slot, slot);                                                 \
        --i;                                                                   \
      }                                                                        \
    }                                                                          \
    hashtbl_reset_available(tbl);                                              \
    FREE_(slot, SLOT_SIZE_);                                                   \
  }

#define HASHTBL_INTERNAL_REHASH_AND_GROW_IF_NECESSARY_META(                    \
    NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,         \
    SLOT_ALIGN_, SLOT_SIZE_)                                                   \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(                         \
      hashtbl_rehash_and_grow_if_necessary, NAME_)(struct hashtbl * tbl) {     \
    if (tbl->capacity == 0) {                                                  \
      tbl = HASHTBL_INTERNAL_META(hashtbl_resize, NAME_)(tbl, 1);              \
    } else if (tbl->capacity > GROUP_WIDTH &&                                  \
               tbl->size * 32ull <= tbl->capacity * 25ull) {                   \
      HASHTBL_INTERNAL_META(hashtbl_drop_deletes_without_resize, NAME_)(tbl);  \
    } else {                                                                   \
      tbl = HASHTBL_INTERNAL_META(hashtbl_resize,                              \
                                  NAME_)(tbl, tbl->capacity * 2 + 1);          \
    }                                                                          \
    return tbl;                                                                \
  }

#define HASHTBL_INTERNAL_PREPARE_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,  \
                                             FREE_, GET_, HASH_, INIT_, MOVE_, \
                                             SLOT_ALIGN_, SLOT_SIZE_)          \
  static size_t HASHTBL_INTERNAL_META(hashtbl_prepare_insert, NAME_)(          \
      struct hashtbl * *ptbl, size_t hash) {                                   \
    struct hashtbl *tbl = *ptbl;                                               \
    struct find_info target =                                                  \
        hashtbl_find_first_non_full(tbl->ctrl, hash, tbl->capacity);           \
    if (tbl->available == 0 &&                                                 \
        !hashtbl_ctrl_is_deleted(tbl->ctrl[target.offset])) {                  \
      tbl = HASHTBL_INTERNAL_META(hashtbl_rehash_and_grow_if_necessary,        \
                                  NAME_)(tbl);                                 \
      target = hashtbl_find_first_non_full(tbl->ctrl, hash, tbl->capacity);    \
    }                                                                          \
    tbl->size += 1;                                                            \
    tbl->available -= hashtbl_ctrl_is_empty(tbl->ctrl[target.offset]);         \
    hashtbl_set_ctrl(target.offset, h2_hash_as_ctrl(hash), tbl->capacity,      \
                     tbl->ctrl);                                               \
    *ptbl = tbl;                                                               \
    return target.offset;                                                      \
  }

#define HASHTBL_INTERNAL_FIND_OR_PREPARE_INSERT_META(                          \
    NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,         \
    SLOT_ALIGN_, SLOT_SIZE_)                                                   \
  static inline struct prepare_insert HASHTBL_INTERNAL_META(                   \
      hashtbl_find_or_prepare_insert, NAME_)(struct hashtbl * *ptbl,           \
                                             const void *key) {                \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    size_t hash = HASH_(key);                                                  \
    struct hashtbl *tbl = *ptbl;                                               \
    struct probe_seq seq =                                                     \
        hashtbl_probe_seq_start(tbl->ctrl, hash, tbl->capacity);               \
    while (true) {                                                             \
      group_t g = hashtbl_group_new(tbl->ctrl + seq.offset);                   \
      struct bitmask match = hashtbl_group_match(&g, h2_hash(hash));           \
      uint32_t i;                                                              \
      while (bitmask_next(&match, &i)) {                                       \
        size_t idx = hashtbl_probe_seq_offset(&seq, i);                        \
        uint8_t *slot = tbl->slots + idx * SLOT_SIZE_;                         \
        if (EQ_(key, GET_(slot)))                                              \
          return (struct prepare_insert){idx, false};                          \
      }                                                                        \
      if (hashtbl_group_match_empty(&g).mask)                                  \
        break;                                                                 \
      hashtbl_probe_seq_next(&seq);                                            \
      HASHTBL_DCHECK(seq.index <= tbl->capacity, "full table!");               \
    }                                                                          \
    *ptbl = tbl;                                                               \
    return (struct prepare_insert){                                            \
        HASHTBL_INTERNAL_META(hashtbl_prepare_insert, NAME_)(ptbl, hash),      \
        true};                                                                 \
  }

#define HASHTBL_INTERNAL_PRE_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,      \
                                         FREE_, GET_, HASH_, INIT_, MOVE_,     \
                                         SLOT_ALIGN_, SLOT_SIZE_)              \
  static inline void *HASHTBL_INTERNAL_META(hashtbl_pre_insert, NAME_)(        \
      struct hashtbl * tbl, size_t i) {                                        \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    void *dst = tbl->slots + i * SLOT_SIZE_;                                   \
    INIT_(dst);                                                                \
    return GET_(dst);                                                          \
  }

#define HASHTBL_INTERNAL_NEW_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,      \
                                  GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,      \
                                  SLOT_SIZE_)                                  \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(hashtbl_new, NAME_)(     \
      size_t capacity) {                                                       \
    return HASHTBL_INTERNAL_META(hashtbl_alloc_new_table, NAME_)(              \
        hashtbl_normalize_capacity(capacity), 0);                              \
  }

#define HASHTBL_INTERNAL_RESERVE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,  \
                                      GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,  \
                                      SLOT_SIZE_)                              \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(hashtbl_reserve, NAME_)( \
      struct hashtbl * tbl, size_t n) {                                        \
    if (n <= tbl->size + tbl->available)                                       \
      return tbl;                                                              \
                                                                               \
    n = hashtbl_normalize_capacity(hashtbl_growth_to_lower_bound_capacity(n)); \
    tbl = HASHTBL_INTERNAL_META(hashtbl_resize, NAME_)(tbl, n);                \
                                                                               \
    return tbl;                                                                \
  }

#define HASHTBL_INTERNAL_DESTROY_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,  \
                                      GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,  \
                                      SLOT_SIZE_)                              \
  static inline void HASHTBL_INTERNAL_META(hashtbl_destroy,                    \
                                           NAME_)(struct hashtbl * tbl) {      \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    HASHTBL_INTERNAL_META(hashtbl_destroy_slots, NAME_)(tbl);                  \
    FREE_(tbl,                                                                 \
          HASHTBL_INTERNAL_META(hashtbl_alloc_size, NAME_)(tbl->capacity));    \
  }

static inline bool hashtbl_is_empty(const struct hashtbl *tbl) {
  return tbl->size == 0;
}

static inline size_t hashtbl_size(const struct hashtbl *tbl) {
  return tbl->size;
}

static inline size_t hashtbl_capacity(const struct hashtbl *tbl) {
  return tbl->capacity;
}

#define HASHTBL_INTERNAL_CLEAR_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,    \
                                    GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,    \
                                    SLOT_SIZE_)                                \
  static inline void HASHTBL_INTERNAL_META(hashtbl_clear,                      \
                                           NAME_)(struct hashtbl * tbl) {      \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    if (tbl->capacity) {                                                       \
      for (size_t i = 0; i != tbl->capacity; ++i)                              \
        if (hashtbl_ctrl_is_full(tbl->ctrl[i]))                                \
          DEL_(tbl->slots + i * SLOT_SIZE_);                                   \
      tbl->size = 0;                                                           \
      hashtbl_reset_ctrl(tbl->capacity, tbl->ctrl);                            \
      hashtbl_reset_available(tbl);                                            \
    }                                                                          \
    HASHTBL_DCHECK(!tbl->size, "size was still nonzero");                      \
  }

#define HASHTBL_INTERNAL_DEFERRED_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, \
                                              FREE_, GET_, HASH_, INIT_,       \
                                              MOVE_, SLOT_ALIGN_, SLOT_SIZE_)  \
  static inline struct insert HASHTBL_INTERNAL_META(                           \
      hashtbl_deferred_insert, NAME_)(struct hashtbl * *ptbl,                  \
                                      const void *key) {                       \
    struct prepare_insert res = HASHTBL_INTERNAL_META(                         \
        hashtbl_find_or_prepare_insert, NAME_)(ptbl, key);                     \
                                                                               \
    if (res.inserted) {                                                        \
      HASHTBL_INTERNAL_META(hashtbl_pre_insert, NAME_)(*ptbl, res.index);      \
    }                                                                          \
                                                                               \
    return (struct insert){                                                    \
        HASHTBL_INTERNAL_META(hashtbl_const_iter_at, NAME_)(*ptbl, res.index), \
        res.inserted};                                                         \
  }

#define HASHTBL_INTERNAL_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,   \
                                     GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,   \
                                     SLOT_SIZE_)                               \
  static inline struct insert HASHTBL_INTERNAL_META(hashtbl_insert, NAME_)(    \
      struct hashtbl * *ptbl, const void *val) {                               \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    struct prepare_insert res = HASHTBL_INTERNAL_META(                         \
        hashtbl_find_or_prepare_insert, NAME_)(ptbl, val);                     \
                                                                               \
    if (res.inserted) {                                                        \
      void *slot =                                                             \
          HASHTBL_INTERNAL_META(hashtbl_pre_insert, NAME_)(*ptbl, res.index);  \
      COPY_(slot, val);                                                        \
    }                                                                          \
    return (struct insert){                                                    \
        HASHTBL_INTERNAL_META(hashtbl_const_iter_at, NAME_)(*ptbl, res.index), \
        res.inserted};                                                         \
  }

#define HASHTBL_INTERNAL_FIND_HINTED_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,     \
                                          FREE_, GET_, HASH_, INIT_, MOVE_,    \
                                          SLOT_ALIGN_, SLOT_SIZE_)             \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_find_hinted, NAME_)(const struct hashtbl *tbl, const void *key,  \
                                  size_t hash) {                               \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    struct probe_seq seq =                                                     \
        hashtbl_probe_seq_start(tbl->ctrl, hash, tbl->capacity);               \
    while (true) {                                                             \
      group_t g = hashtbl_group_new(tbl->ctrl + seq.offset);                   \
      struct bitmask match = hashtbl_group_match(&g, h2_hash(hash));           \
      uint32_t i;                                                              \
      while (bitmask_next(&match, &i)) {                                       \
        uint8_t *slot =                                                        \
            tbl->slots + hashtbl_probe_seq_offset(&seq, i) * SLOT_SIZE_;       \
        if (EQ_(key, GET_(slot)))                                              \
          return HASHTBL_INTERNAL_META(hashtbl_const_iter_at, NAME_)(          \
              tbl, hashtbl_probe_seq_offset(&seq, i));                         \
      }                                                                        \
      if (hashtbl_group_match_empty(&g).mask)                                  \
        return (struct hashtbl_iterator){0};                                   \
      hashtbl_probe_seq_next(&seq);                                            \
      HASHTBL_DCHECK(seq.index <= tbl->capacity, "full table!");               \
    }                                                                          \
  }

#define HASHTBL_INTERNAL_FIND_HINTED_BY_META(NAME_, GET_, EQ2_, S_)            \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_find_hinted_by_##S_, NAME_)(const struct hashtbl *tbl,           \
                                          const void *key, size_t hash) {      \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    struct probe_seq seq =                                                     \
        hashtbl_probe_seq_start(tbl->ctrl, hash, tbl->capacity);               \
    while (true) {                                                             \
      group_t g = hashtbl_group_new(tbl->ctrl + seq.offset);                   \
      struct bitmask match = hashtbl_group_match(&g, h2_hash(hash));           \
      uint32_t i;                                                              \
      while (bitmask_next(&match, &i)) {                                       \
        uint8_t *slot = tbl->slots + hashtbl_probe_seq_offset(&seq, i) *       \
                                         sizeof(this_entry_type_);             \
        if (EQ2_(key, GET_(slot)))                                             \
          return HASHTBL_INTERNAL_META(hashtbl_const_iter_at, NAME_)(          \
              tbl, hashtbl_probe_seq_offset(&seq, i));                         \
      }                                                                        \
      if (hashtbl_group_match_empty(&g).mask)                                  \
        return (struct hashtbl_iterator){0};                                   \
      hashtbl_probe_seq_next(&seq);                                            \
      HASHTBL_DCHECK(seq.index <= tbl->capacity, "full table!");               \
    }                                                                          \
  }

#define HASHTBL_INTERNAL_FIND_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,     \
                                   GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,     \
                                   SLOT_SIZE_)                                 \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_find, NAME_)(const struct hashtbl *tbl, const void *key) {       \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    return HASHTBL_INTERNAL_META(hashtbl_find_hinted, NAME_)(tbl, key,         \
                                                             HASH_(key));      \
  }

#define HASHTBL_INTERNAL_FIND_BY_META(NAME_, HASH2_, S_)                       \
  static inline struct hashtbl_iterator HASHTBL_INTERNAL_META(                 \
      hashtbl_find_by_##S_, NAME_)(const struct hashtbl *tbl,                  \
                                   const void *key) {                          \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    return HASHTBL_INTERNAL_META(hashtbl_find_hinted_by_##S_,                  \
                                 NAME_)(tbl, key, HASH2_(key));                \
  }

#define HASHTBL_INTERNAL_ERASE_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, \
                                       GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_, \
                                       SLOT_SIZE_)                             \
  static inline void HASHTBL_INTERNAL_META(hashtbl_erase_at, NAME_)(           \
      struct hashtbl_iterator it) {                                            \
    MAYBEUNUSED typedef HASHTBL_META(key_t, NAME_) this_key_type_;             \
    MAYBEUNUSED typedef HASHTBL_META(entry_t, NAME_) this_entry_type_;         \
                                                                               \
    HASHTBL_ASSERT_IS_FULL(it.ctrl);                                           \
    DEL_(it.slot);                                                             \
    hashtbl_erase_meta_only(it);                                               \
  }

#define HASHTBL_INTERNAL_ERASE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,    \
                                    GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,    \
                                    SLOT_SIZE_)                                \
  static inline bool HASHTBL_INTERNAL_META(hashtbl_erase, NAME_)(              \
      struct hashtbl * tbl, const void *key) {                                 \
    struct hashtbl_iterator it =                                               \
        HASHTBL_INTERNAL_META(hashtbl_find, NAME_)(tbl, key);                  \
    if (it.slot == NULL)                                                       \
      return false;                                                            \
    HASHTBL_INTERNAL_META(hashtbl_erase_at, NAME_)(it);                        \
    return true;                                                               \
  }

#define HASHTBL_INTERNAL_ERASE_BY_META(NAME_, S_)                              \
  static inline bool HASHTBL_INTERNAL_META(hashtbl_erase_by_##S_, NAME_)(      \
      struct hashtbl * tbl, const void *key) {                                 \
    struct hashtbl_iterator it =                                               \
        HASHTBL_INTERNAL_META(hashtbl_find_by_##S_, NAME_)(tbl, key);          \
    if (it.slot == NULL)                                                       \
      return false;                                                            \
    HASHTBL_INTERNAL_META(hashtbl_erase_at, NAME_)(it);                        \
    return true;                                                               \
  }

#define HASHTBL_INTERNAL_REHASH_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,   \
                                     GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,   \
                                     SLOT_SIZE_)                               \
  static inline struct hashtbl *HASHTBL_INTERNAL_META(hashtbl_rehash, NAME_)(  \
      struct hashtbl * tbl, size_t n) {                                        \
    if (n == 0 && tbl->capacity == 0)                                          \
      return tbl;                                                              \
    if (n == 0 && tbl->size == 0) {                                            \
      HASHTBL_INTERNAL_META(hashtbl_destroy_slots, NAME_)(tbl);                \
      return tbl;                                                              \
    }                                                                          \
                                                                               \
    size_t m = hashtbl_normalize_capacity(                                     \
        n | hashtbl_growth_to_lower_bound_capacity(tbl->size));                \
    if (n == 0 || m > tbl->capacity)                                           \
      return HASHTBL_INTERNAL_META(hashtbl_resize, NAME_)(tbl, m);             \
    return tbl;                                                                \
  }

#define HASHTBL_INTERNAL_CONTAINS_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, \
                                       GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_, \
                                       SLOT_SIZE_)                             \
  static inline bool HASHTBL_INTERNAL_META(hashtbl_contains, NAME_)(           \
      const struct hashtbl *tbl, const void *key) {                            \
    return HASHTBL_INTERNAL_META(hashtbl_find, NAME_)(tbl, key).slot != NULL;  \
  }

#define HASHTBL_INTERNAL_CONTAINS_BY_META(NAME_, S_)                           \
  static inline bool HASHTBL_INTERNAL_META(hashtbl_contains_by_##S_, NAME_)(   \
      const struct hashtbl *tbl, const void *key) {                            \
    return HASHTBL_INTERNAL_META(hashtbl_find_by_##S_, NAME_)(tbl, key)        \
               .slot != NULL;                                                  \
  }

#define HASHTBL_INTERNAL_COMMON_FUNCTIONS_META(NAME_, ALLOC_, COPY_, DEL_,     \
                                               EQ_, FREE_, GET_, HASH_, INIT_, \
                                               MOVE_, SLOT_ALIGN_, SLOT_SIZE_) \
  HASHTBL_INTERNAL_ALLOC_SIZE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,     \
                                   GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,     \
                                   SLOT_SIZE_)                                 \
  HASHTBL_INTERNAL_ITER_SKIP_EMPTY_OR_DELETED_META(                            \
      NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,       \
      SLOT_ALIGN_, SLOT_SIZE_)                                                 \
  HASHTBL_INTERNAL_ITER_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,  \
                                HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)  \
  HASHTBL_INTERNAL_ITER_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,     \
                             HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)     \
  HASHTBL_INTERNAL_CONST_ITER_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,  \
                                      GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,  \
                                      SLOT_SIZE_)                              \
  HASHTBL_INTERNAL_CONST_ITER_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,     \
                                   GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,     \
                                   SLOT_SIZE_)                                 \
  HASHTBL_INTERNAL_ITER_GET_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, \
                                 HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_) \
  HASHTBL_INTERNAL_ITER_NEXT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,      \
                                  GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,      \
                                  SLOT_SIZE_)                                  \
  HASHTBL_INTERNAL_ALLOC_NEW_TABLE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,       \
                                        FREE_, GET_, HASH_, INIT_, MOVE_,      \
                                        SLOT_ALIGN_, SLOT_SIZE_)               \
  HASHTBL_INTERNAL_DESTROY_SLOTS_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,  \
                                      GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,  \
                                      SLOT_SIZE_)                              \
  HASHTBL_INTERNAL_RESIZE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,   \
                               HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)   \
  HASHTBL_INTERNAL_DROP_DELETES_WITHOUT_RESIZE_META(                           \
      NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,       \
      SLOT_ALIGN_, SLOT_SIZE_)                                                 \
  HASHTBL_INTERNAL_REHASH_AND_GROW_IF_NECESSARY_META(                          \
      NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,       \
      SLOT_ALIGN_, SLOT_SIZE_)                                                 \
  HASHTBL_INTERNAL_PREPARE_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, \
                                       GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_, \
                                       SLOT_SIZE_)                             \
  HASHTBL_INTERNAL_FIND_OR_PREPARE_INSERT_META(NAME_, ALLOC_, COPY_, DEL_,     \
                                               EQ_, FREE_, GET_, HASH_, INIT_, \
                                               MOVE_, SLOT_ALIGN_, SLOT_SIZE_) \
  HASHTBL_INTERNAL_PRE_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,     \
                                   GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,     \
                                   SLOT_SIZE_)                                 \
  HASHTBL_INTERNAL_NEW_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,      \
                            HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)      \
  HASHTBL_INTERNAL_RESERVE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,  \
                                HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)  \
  HASHTBL_INTERNAL_DESTROY_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,  \
                                HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)  \
  HASHTBL_INTERNAL_CLEAR_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,    \
                              HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)    \
  HASHTBL_INTERNAL_DEFERRED_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,       \
                                        FREE_, GET_, HASH_, INIT_, MOVE_,      \
                                        SLOT_ALIGN_, SLOT_SIZE_)               \
  HASHTBL_INTERNAL_INSERT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,   \
                               HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)   \
  HASHTBL_INTERNAL_FIND_HINTED_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_,    \
                                    GET_, HASH_, INIT_, MOVE_, SLOT_ALIGN_,    \
                                    SLOT_SIZE_)                                \
  HASHTBL_INTERNAL_FIND_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,     \
                             HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)     \
  HASHTBL_INTERNAL_ERASE_AT_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, \
                                 HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_) \
  HASHTBL_INTERNAL_ERASE_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,    \
                              HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)    \
  HASHTBL_INTERNAL_REHASH_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_,   \
                               HASH_, INIT_, MOVE_, SLOT_ALIGN_, SLOT_SIZE_)   \
  HASHTBL_INTERNAL_CONTAINS_META(NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, \
                                 HASH_, INIT_, MOVE_, SLOT_ALIGN_,             \
                                 SLOT_SIZE_);

#define HASHTBL_COMMON_FUNCTIONS_META(SLOT_TYPE_, KEY_TYPE_, NAME_)            \
                                                                               \
  typedef struct NAME_ {                                                       \
    struct hashtbl raw;                                                        \
  } HASHTBL_META(t, NAME_);                                                    \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct NAME_ *HASHTBL_META(new, NAME_)(size_t n) {             \
    return (struct NAME_ *)HASHTBL_INTERNAL_META(hashtbl_new, NAME_)(n);       \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline void HASHTBL_META(destroy, NAME_)(struct NAME_ * tbl) {        \
    HASHTBL_INTERNAL_META(hashtbl_destroy, NAME_)(&tbl->raw);                  \
  }                                                                            \
                                                                               \
  typedef struct HASHTBL_META(iterator, NAME_) {                               \
    struct hashtbl_iterator raw;                                               \
  } HASHTBL_META(iter_t, NAME_);                                               \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(iterator, NAME_)                           \
      HASHTBL_META(iter, NAME_)(struct NAME_ * tbl) {                          \
    return (struct HASHTBL_META(iterator, NAME_)){                             \
        HASHTBL_INTERNAL_META(hashtbl_iter, NAME_)(&tbl->raw)};                \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline SLOT_TYPE_ *HASHTBL_META(iter_get, NAME_)(                     \
      const struct HASHTBL_META(iterator, NAME_) * iter) {                     \
    return (SLOT_TYPE_ *)HASHTBL_INTERNAL_META(hashtbl_iter_get,               \
                                               NAME_)(&iter->raw);             \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline SLOT_TYPE_ *HASHTBL_META(iter_next, NAME_)(                    \
      struct HASHTBL_META(iterator, NAME_) * iter) {                           \
    return (SLOT_TYPE_ *)HASHTBL_INTERNAL_META(hashtbl_iter_next,              \
                                               NAME_)(&iter->raw);             \
  }                                                                            \
                                                                               \
  struct HASHTBL_META(const_iterator, NAME_) {                                 \
    struct hashtbl_iterator raw;                                               \
  };                                                                           \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(const_iterator, NAME_)                     \
      HASHTBL_META(const_iter, NAME_)(struct NAME_ * tbl) {                    \
    return (struct HASHTBL_META(const_iterator, NAME_)){                       \
        HASHTBL_INTERNAL_META(hashtbl_const_iter, NAME_)(&tbl->raw)};          \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline const SLOT_TYPE_ *HASHTBL_META(const_iter_get, NAME_)(         \
      const struct HASHTBL_META(iterator, NAME_) * iter) {                     \
    return (const SLOT_TYPE_ *)HASHTBL_INTERNAL_META(hashtbl_iter_get,         \
                                                     NAME_)(&iter->raw);       \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline const SLOT_TYPE_ *HASHTBL_META(const_iter_next, NAME_)(        \
      struct HASHTBL_META(iterator, NAME_) * iter) {                           \
    return (const SLOT_TYPE_ *)HASHTBL_INTERNAL_META(hashtbl_iter_next,        \
                                                     NAME_)(&iter->raw);       \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct NAME_ *HASHTBL_META(reserve, NAME_)(struct NAME_ * tbl, \
                                                           size_t n) {         \
    return (struct NAME_ *)HASHTBL_INTERNAL_META(hashtbl_reserve,              \
                                                 NAME_)(&tbl->raw, n);         \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct NAME_ *HASHTBL_META(rehash, NAME_)(struct NAME_ * tbl,  \
                                                          size_t n) {          \
    return (struct NAME_ *)HASHTBL_INTERNAL_META(hashtbl_rehash,               \
                                                 NAME_)(&tbl->raw, n);         \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline bool HASHTBL_META(is_empty, NAME_)(struct NAME_ * tbl) {       \
    return hashtbl_is_empty(&tbl->raw);                                        \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline size_t HASHTBL_META(size, NAME_)(const struct NAME_ *tbl) {    \
    return hashtbl_size(&tbl->raw);                                            \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline size_t HASHTBL_META(capacity,                                  \
                                    NAME_)(const struct NAME_ *tbl) {          \
    return hashtbl_capacity(&tbl->raw);                                        \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline void HASHTBL_META(clear, NAME_)(struct NAME_ * tbl) {          \
    return HASHTBL_INTERNAL_META(hashtbl_clear, NAME_)(&tbl->raw);             \
  }                                                                            \
                                                                               \
  typedef struct HASHTBL_META(insert, NAME_) {                                 \
    struct HASHTBL_META(iterator, NAME_) iter;                                 \
    bool inserted;                                                             \
  } HASHTBL_META(insert_t, NAME_);                                             \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(insert, NAME_) HASHTBL_META(               \
      deferred_insert, NAME_)(struct NAME_ * *ptbl, const KEY_TYPE_ *key) {    \
    struct hashtbl *tbl = &((*ptbl)->raw);                                     \
    struct insert insert =                                                     \
        HASHTBL_INTERNAL_META(hashtbl_deferred_insert, NAME_)(&tbl, key);      \
    *ptbl = (struct NAME_ *)tbl;                                               \
    return (struct HASHTBL_META(insert, NAME_)){{insert.iter},                 \
                                                insert.inserted};              \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(insert, NAME_) HASHTBL_META(               \
      insert, NAME_)(struct NAME_ * *ptbl, const KEY_TYPE_ *key) {             \
    struct hashtbl *tbl = &((*ptbl)->raw);                                     \
    struct insert insert =                                                     \
        HASHTBL_INTERNAL_META(hashtbl_insert, NAME_)(&tbl, key);               \
    *ptbl = (struct NAME_ *)tbl;                                               \
    return (struct HASHTBL_META(insert, NAME_)){{insert.iter},                 \
                                                insert.inserted};              \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(iterator, NAME_)                           \
      HASHTBL_META(find_hinted, NAME_)(const struct NAME_ *tbl,                \
                                       const KEY_TYPE_ *key, size_t hash) {    \
    return (struct HASHTBL_META(iterator, NAME_)){HASHTBL_INTERNAL_META(       \
        hashtbl_find_hinted, NAME_)(&tbl->raw, key, hash)};                    \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(const_iterator, NAME_)                     \
      HASHTBL_META(const_find_hinted, NAME_)(                                  \
          const struct NAME_ *tbl, const KEY_TYPE_ *key, size_t hash) {        \
    return (struct HASHTBL_META(const_iterator, NAME_)){HASHTBL_INTERNAL_META( \
        hashtbl_find_hinted, NAME_)(&tbl->raw, key, hash)};                    \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(iterator, NAME_) HASHTBL_META(             \
      find, NAME_)(const struct NAME_ *tbl, const KEY_TYPE_ *key) {            \
    return (struct HASHTBL_META(iterator, NAME_)){                             \
        HASHTBL_INTERNAL_META(hashtbl_find, NAME_)(&tbl->raw, key)};           \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(const_iterator, NAME_) HASHTBL_META(       \
      const_find, NAME_)(const struct NAME_ *tbl, const KEY_TYPE_ *key) {      \
    return (struct HASHTBL_META(const_iterator, NAME_)){                       \
        HASHTBL_INTERNAL_META(hashtbl_find, NAME_)(&tbl->raw, key)};           \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline bool HASHTBL_META(contains, NAME_)(const struct NAME_ *tbl,    \
                                                   const KEY_TYPE_ *key) {     \
    return HASHTBL_INTERNAL_META(hashtbl_contains, NAME_)(&tbl->raw, key);     \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline void HASHTBL_META(erase_at, NAME_)(                            \
      struct HASHTBL_META(iterator, NAME_) iter) {                             \
    HASHTBL_INTERNAL_META(hashtbl_erase_at, NAME_)(iter.raw);                  \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline bool HASHTBL_META(erase, NAME_)(struct NAME_ * tbl,            \
                                                const KEY_TYPE_ *key) {        \
    return HASHTBL_INTERNAL_META(hashtbl_erase, NAME_)(&tbl->raw, key);        \
  }

#define HASHTBL_INTERNAL_LOOKUP_FUNCTIONS_META(NAME_, GET_, EQ2_, HASH2_, S_)  \
  HASHTBL_INTERNAL_FIND_HINTED_BY_META(NAME_, GET_, EQ2_, S_)                  \
  HASHTBL_INTERNAL_FIND_BY_META(NAME_, HASH2_, S_)                             \
  HASHTBL_INTERNAL_ERASE_BY_META(NAME_, S_)                                    \
  HASHTBL_INTERNAL_CONTAINS_BY_META(NAME_, S_)

#define HASHTBL_LOOKUP_FUNCTIONS_META(NAME_, S_, KEY_TYPE_)                    \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(iterator, NAME_)                           \
      HASHTBL_META(find_hinted_by_##S_, NAME_)(                                \
          const struct NAME_ *tbl, const KEY_TYPE_ *key, size_t hash) {        \
    return (struct HASHTBL_META(iterator, NAME_)){HASHTBL_INTERNAL_META(       \
        hashtbl_find_hinted_by_##S_, NAME_)(&tbl->raw, key, hash)};            \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(const_iterator, NAME_)                     \
      HASHTBL_META(const_find_hinted_by_##S_, NAME_)(                          \
          const struct NAME_ *tbl, const KEY_TYPE_ *key, size_t hash) {        \
    return (struct HASHTBL_META(const_iterator, NAME_)){HASHTBL_INTERNAL_META( \
        hashtbl_find_hinted_by_##S_, NAME_)(&tbl->raw, key, hash)};            \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(iterator, NAME_) HASHTBL_META(             \
      find_by_##S_, NAME_)(const struct NAME_ *tbl, const KEY_TYPE_ *key) {    \
    return (struct HASHTBL_META(iterator, NAME_)){                             \
        HASHTBL_INTERNAL_META(hashtbl_find_by_##S_, NAME_)(&tbl->raw, key)};   \
  }                                                                            \
                                                                               \
  MAYBEUNUSED                                                                  \
  static inline struct HASHTBL_META(const_iterator, NAME_)                     \
      HASHTBL_META(const_find_by_##S_, NAME_)(const struct NAME_ *tbl,         \
                                              const KEY_TYPE_ *key) {          \
    return (struct HASHTBL_META(const_iterator, NAME_)){                       \
        HASHTBL_INTERNAL_META(hashtbl_find_by_##S_, NAME_)(&tbl->raw, key)};   \
  }                                                                            \
  MAYBEUNUSED                                                                  \
  static inline bool HASHTBL_META(contains_by_##S_, NAME_)(                    \
      const struct NAME_ *tbl, const KEY_TYPE_ *key) {                         \
    return HASHTBL_INTERNAL_META(hashtbl_contains_by_##S_, NAME_)(&tbl->raw,   \
                                                                  key);        \
  }                                                                            \
  MAYBEUNUSED                                                                  \
  static inline bool HASHTBL_META(erase_by_##S_, NAME_)(                       \
      struct NAME_ * tbl, const KEY_TYPE_ *key) {                              \
    return HASHTBL_INTERNAL_META(hashtbl_erase_by_##S_, NAME_)(&tbl->raw,      \
                                                               key);           \
  }

#define HASHSET_NEW_KIND(NAME_, KEY_TYPE_, KEY_ALIGN_, ALLOC_, COPY_, DEL_,    \
                         EQ_, FREE_, GET_, HASH_, INIT_, MOVE_)                \
  typedef KEY_TYPE_ HASHTBL_META(key_t, NAME_);                                \
  typedef KEY_TYPE_ HASHTBL_META(entry_t, NAME_);                              \
  HASHTBL_INTERNAL_COMMON_FUNCTIONS_META(NAME_, ALLOC_, COPY_, DEL_, EQ_,      \
                                         FREE_, GET_, HASH_, INIT_, MOVE_,     \
                                         KEY_ALIGN_, sizeof(KEY_TYPE_))        \
  HASHTBL_COMMON_FUNCTIONS_META(KEY_TYPE_, KEY_TYPE_, NAME_);

#define HASHMAP_NEW_KIND(NAME_, KEY_TYPE_, VAL_TYPE_, ALIGN_, ALLOC_, COPY_,   \
                         DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_)          \
  typedef KEY_TYPE_ HASHTBL_META(key_t, NAME_);                                \
  typedef VAL_TYPE_ HASHTBL_META(val_t, NAME_);                                \
  typedef struct {                                                             \
    HASHTBL_META(key_t, NAME_) key;                                            \
    HASHTBL_META(val_t, NAME_) val;                                            \
  } HASHTBL_META(entry_t, NAME_);                                              \
  HASHTBL_INTERNAL_COMMON_FUNCTIONS_META(                                      \
      NAME_, ALLOC_, COPY_, DEL_, EQ_, FREE_, GET_, HASH_, INIT_, MOVE_,       \
      ALIGN_, sizeof(HASHTBL_META(entry_t, NAME_)))                            \
  HASHTBL_COMMON_FUNCTIONS_META(HASHTBL_META(entry_t, NAME_),                  \
                                HASHTBL_META(key_t, NAME_), NAME_);

#define HASHTBL_NEW_LOOKUP_KIND(NAME_, KEY_TYPE_, NAME2_, EQ_, GET_, HASH_)    \
  HASHTBL_INTERNAL_LOOKUP_FUNCTIONS_META(NAME_, GET_, EQ_, HASH_, NAME2_)      \
  HASHTBL_LOOKUP_FUNCTIONS_META(NAME_, NAME2_, KEY_TYPE_)

MAYBEUNUSED
static inline void *hashtbl_default_alloc(size_t size) {
  void *p = malloc(size);
  HASHTBL_CHECK(p != NULL, "malloc() returned null");
  return p;
}

MAYBEUNUSED
static inline void hashtbl_default_free(void *array, size_t size) {
  (void)size;
  free(array);
}

MAYBEUNUSED
static inline uint64_t hashtbl_default_hash(const void *data, size_t size) {
  return komihash(data, size, 0);
}

#define DEFAULT_ALLOC(size) hashtbl_default_alloc((size))
#define DEFAULT_COPY(dst, src)                                                 \
  memcpy((void *)(dst), (void *)(src), sizeof(this_entry_type_))
#define DEFAULT_DEL(x) ((void)(x))
#define DEFAULT_EQ(x, y) (memcmp((x), (y), sizeof(this_key_type_)) == 0)
#define DEFAULT_FREE(x, size) hashtbl_default_free((x), (size))
#define DEFAULT_GET(x) ((this_entry_type_ *)(x))
#define DEFAULT_HASH(x) hashtbl_default_hash((x), sizeof(this_key_type_))
#define DEFAULT_INIT(x) ((void)(x))
#define DEFAULT_MOVE(dst, src)                                                 \
  memcpy((void *)(dst), (void *)(src), sizeof(this_entry_type_))

#define HASHMAP_NEW_KIND_WITH_DEFAULTS(NAME_, KEY_TYPE_, VAL_TYPE_, ALIGN_)    \
  HASHMAP_NEW_KIND(NAME_, KEY_TYPE_, VAL_TYPE_, ALIGN_, DEFAULT_ALLOC,         \
                   DEFAULT_COPY, DEFAULT_DEL, DEFAULT_EQ, DEFAULT_FREE,        \
                   DEFAULT_GET, DEFAULT_HASH, DEFAULT_INIT, DEFAULT_MOVE)

#define HASHSET_NEW_KIND_WITH_DEFAULTS(NAME_, KEY_TYPE_, ALIGN_)               \
  HASHSET_NEW_KIND(NAME_, KEY_TYPE_, ALIGN_, DEFAULT_ALLOC, DEFAULT_COPY,      \
                   DEFAULT_DEL, DEFAULT_EQ, DEFAULT_FREE, DEFAULT_GET,         \
                   DEFAULT_HASH, DEFAULT_INIT, DEFAULT_MOVE)

#define HASHTBL_NEW_LOOKUP_KIND_WITH_DEFAULTS(NAME_, KEY_TYPE_, NAME2_, EQ_,   \
                                              HASH_)                           \
  HASHTBL_NEW_LOOKUP_KIND(NAME_, KEY_TYPE_, NAME2_, EQ_, DEFAULT_GET, HASH_)

#endif
