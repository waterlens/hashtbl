#include <assert.h>
#include <stdint.h>

#include "hashtbl.h"

#define INT64_EQ(a, b) (*(const int64_t *)(a) == (b)->key)
#define INT64_HASH(x)                                                          \
  (hashtbl_default_hash((const int64_t *)(x), sizeof(int64_t)))
HASHMAP_NEW_KIND(i64map, int64_t, int64_t, 8, DEFAULT_ALLOC, DEFAULT_COPY,
                 DEFAULT_DEL, INT64_EQ, DEFAULT_FREE, DEFAULT_GET, INT64_HASH,
                 DEFAULT_INIT, DEFAULT_MOVE)

int main() {
  i64map_t *map = i64map_new(10);

  size_t s = i64map_size(map);
  assert(s == 0);

  int64_t keys[][2] = {
      {1, 2}, {2, 3}, {3, 4}, {4, 5},  {5, 6},
      {6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11},
  };

  for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
    i64map_insert_t ins = i64map_insert(&map, &keys[i][0]);
    i64map_entry_t *entry = i64map_iter_get(&ins.iter);
    entry->key = keys[i][0];
    entry->val = keys[i][1];
  }

  s = i64map_size(map);
  assert(s == 10);

  for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
    i64map_iter_t iter = i64map_find(map, &keys[i][0]);
    i64map_entry_t *entry = i64map_iter_get(&iter);
    assert(entry->key == keys[i][0]);
    assert(entry->val == keys[i][1]);
  }

  i64map_destroy(map);
  return 0;
}