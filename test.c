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
  size_t MAX_SIZE = 1000000;

  i64map_t *map = i64map_new(10);

  size_t s = i64map_size(map);
  assert(s == 0);

  for (size_t i = 0; i < MAX_SIZE; i++) {
    int64_t key = i;
    i64map_insert_t ins = i64map_deferred_insert(&map, &key);
    i64map_entry_t *entry = i64map_iter_get(&ins.iter);
    entry->key = key;
    entry->val = MAX_SIZE - i;
  }

  s = i64map_size(map);
  assert(s == MAX_SIZE);

  for (size_t i = 0; i < MAX_SIZE; i++) {
    int64_t key = i;
    i64map_iter_t it = i64map_find(map, &key);
    i64map_entry_t *entry = i64map_iter_get(&it);
    assert(entry->key == (int64_t)key);
    assert(entry->val == (int64_t)(MAX_SIZE - i));
  }

  i64map_iter_t it = i64map_iter(map);

  size_t n = 0;
  for (;; n++) {
    i64map_entry_t *entry = i64map_iter_get(&it);
    if (entry == NULL)
      break;
    assert(entry->key + entry->val == (int64_t)MAX_SIZE);
    i64map_iter_next(&it);
  }

  assert(n == MAX_SIZE);

  i64map_destroy(map);
  return 0;
}