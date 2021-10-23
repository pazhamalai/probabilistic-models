package de.tum.in.probmodels.util;

import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;

// This class is a wrapper around Three level map written for convenience and readability.
public class StateActionSuccessorMap<V> {
    private final Int2ObjectMap<Int2ObjectMap<Int2ObjectMap<V>>> map = new Int2ObjectOpenHashMap<>();

    public void set(int state, int action, int successor, V value) {

        initMap(state, action);
        map.get(state).get(action).put(successor, value);
    }

    public V get(int state, int action , int successor) {
        return map.get(state).get(action).get(successor);
    }

    // Initializes empty hash map's if it's null
    private void initMap(int state, int action) {
        Int2ObjectMap<Int2ObjectMap<V>> stateValues = map.getOrDefault(state, null);

        if (stateValues == null) {
            stateValues = new Int2ObjectOpenHashMap<>();
            map.put(state, stateValues);
        }

        Int2ObjectMap<V> actionValues = stateValues.getOrDefault(action, null);

        if (actionValues == null) {
            actionValues = new Int2ObjectOpenHashMap<>();
            stateValues.put(action, actionValues);
        }
    }
}
