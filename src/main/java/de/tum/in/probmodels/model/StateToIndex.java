package de.tum.in.probmodels.model;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2IntMap;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;

public class StateToIndex<S> {
  private final Object2IntMap<S> stateMap = new Object2IntOpenHashMap<>();
  private final Int2ObjectMap<S> indexMap = new Int2ObjectOpenHashMap<>();

  public StateToIndex() {
    stateMap.defaultReturnValue(-1);
  }

  public void addState(S state, int stateId) {
    assert !indexMap.containsKey(stateId);
    stateMap.put(state, stateId);
    indexMap.put(stateId, state);
  }

  public int getStateId(S state) {
    return stateMap.getInt(state);
  }

  public boolean check(int stateId) {
    return indexMap.containsKey(stateId)
        && stateMap.getInt(indexMap.get(stateId)) == stateId;
  }

  public boolean contains(S state) {
    return stateMap.containsKey(state);
  }

  public S getState(int stateId) {
    return indexMap.get(stateId);
  }

  public int size() {
    return indexMap.size();
  }
}
