package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import java.util.BitSet;
import java.util.Iterator;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntUnaryOperator;

public class Distribution implements Iterable<Int2DoubleMap.Entry> {
  private final Int2DoubleMap map;
  private final NatBitSet support = NatBitSets.set();

  public Distribution() {
    map = new Int2DoubleOpenHashMap();
    map.defaultReturnValue(0.0);
  }

  public Distribution(int key, double value) {
    map = new Int2DoubleOpenHashMap(1);
    map.defaultReturnValue(0.0);
    map.put(key, value);
    support.set(key);
  }

  public void add(int j, double prob) {
    double old = map.getOrDefault(j, -1.0d);
    //noinspection FloatingPointEquality
    if (old == -1.0d) {
      map.put(j, prob);
      support.set(j);
    } else {
      double newValue = prob + old;
      if (newValue <= 0) {
        support.clear(j);
        map.remove(j);
      } else {
        map.put(j, newValue);
      }
    }
  }

  public void set(int j, double prob) {
    // Intentionally do a precise comparison since even the smallest probability changes the
    // underlying graph
    // -> Important for component (SCC/EC) analysis
    if (prob == 0.0d) {
      map.remove(j);
      support.clear(j);
    } else {
      map.put(j, prob);
      support.set(j);
    }
  }

  public double get(int j) {
    return map.get(j);
  }

  public boolean contains(int j) {
    return support.contains(j);
  }

  public NatBitSet support() {
    assert support.equals(map.keySet());
    return support;
  }

  @Override
  public Iterator<Int2DoubleMap.Entry> iterator() {
    return map.int2DoubleEntrySet().iterator();
  }

  @Deprecated
  public Iterator<Map.Entry<Integer, Double>> objectIterator() {
    return map.entrySet().iterator();
  }

  public boolean isEmpty() {
    assert support.isEmpty() == map.isEmpty() : support + " " + map;
    return map.isEmpty();
  }

  public int size() {
    assert support.size() == map.size() : support + " " + map;
    return map.size();
  }

  public double sum() {
    double d = 0.0;
    DoubleIterator iterator = map.values().iterator();
    while (iterator.hasNext()) {
      d += iterator.nextDouble();
    }
    return d;
  }

  public Distribution scale() {
    double total = sum();
    assert total > 0.0d;
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      entry.setValue(entry.getDoubleValue() / total);
    }
    return this;
  }

  public double sumWeighted(IntToDoubleFunction f) {
    double d = 0.0;
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      d += f.applyAsDouble(entry.getIntKey()) * entry.getDoubleValue();
    }
    return d;
  }

  public double sumWeightedExceptJacobi(IntToDoubleFunction f, int state) {
    // TODO Awkward name: This is not only sum but also rescale
    double sum = 0.0d;
    double weight = 0.0d;
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      int s = entry.getIntKey();
      if (s != state) {
        double probability = entry.getDoubleValue();
        sum += f.applyAsDouble(s) * probability;
        weight += probability;
      }
    }
    return weight == 0.0d ? 0.0d : sum / weight;
  }

  public Distribution map(IntUnaryOperator map) {
    Distribution distribution = new Distribution();
    for (Int2DoubleMap.Entry entry : this.map.int2DoubleEntrySet()) {
      int key = map.applyAsInt(entry.getIntKey());
      if (key >= 0) {
        distribution.add(key, entry.getDoubleValue());
      }
    }
    return distribution;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Distribution)) {
      return false;
    }
    Distribution other = (Distribution) o;

    if (!support.equals(other.support)) {
      return false;
    }
    for (Int2DoubleMap.Entry entry : this.map.int2DoubleEntrySet()) {
      if (!Util.isEqual(entry.getDoubleValue(),
          other.map.getOrDefault(entry.getIntKey(), -1))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    return map.keySet().hashCode();
  }

  @Override
  public String toString() {
    return map.toString();
  }

  public boolean isSubsetOf(BitSet set) {
    if (size() > set.size()) {
      return false;
    }
    return NatBitSets.asSet(set).containsAll(support);
  }

  public boolean containsOneOf(BitSet set) {
    if (set.isEmpty()) {
      return true;
    }
    return support.intersects(NatBitSets.asSet(set));
  }

  public boolean containsOneOf(NatBitSet set) {
    if (set.isEmpty()) {
      return true;
    }
    return support.intersects(set);
  }

  public void forEach(BiConsumer<Integer, Double> action) {
    map.forEach(action);
  }
}
