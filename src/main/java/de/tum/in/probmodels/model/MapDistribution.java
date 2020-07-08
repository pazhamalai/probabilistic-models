package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.util.Sample;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntUnaryOperator;

public class MapDistribution implements Distribution {
  private final Int2DoubleMap map;
  private final NatBitSet support;
  private int lazyHash = 0;

  @SuppressWarnings("AssignmentOrReturnOfFieldWithMutableType")
  MapDistribution(Int2DoubleMap map, NatBitSet support) {
    this.map = map;
    this.support = support;
    this.map.defaultReturnValue(0.0);
  }

  MapDistribution(int key, double value) {
    map = new Int2DoubleOpenHashMap(1);
    map.defaultReturnValue(0.0);
    map.put(key, value);
    support = NatBitSets.singleton(key);
  }

  @Override
  public double get(int j) {
    return map.get(j);
  }

  @Override
  public boolean contains(int j) {
    return support.contains(j);
  }

  @SuppressWarnings("AssignmentOrReturnOfFieldWithMutableType")
  @Override
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

  @Override
  public boolean isEmpty() {
    assert support.isEmpty() == map.isEmpty() : support + " " + map;
    return map.isEmpty();
  }

  @Override
  public int size() {
    assert support.size() == map.size() : support + " " + map;
    return map.size();
  }

  @Override
  public double sum() {
    double d = 0.0;
    DoubleIterator iterator = map.values().iterator();
    while (iterator.hasNext()) {
      d += iterator.nextDouble();
    }
    return d;
  }

  @Override
  public double sumWeighted(double[] array) {
    double d = 0.0;
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      d += array[entry.getIntKey()] * entry.getDoubleValue();
    }
    return d;
  }

  @Override
  public double sumWeighted(IntToDoubleFunction f) {
    double d = 0.0;
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      d += f.applyAsDouble(entry.getIntKey()) * entry.getDoubleValue();
    }
    return d;
  }

  @Override
  public double sumWeightedExceptJacobi(IntToDoubleFunction f, int state) {
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

  @Override
  public int sample() {
    return Sample.sample(map);
  }

  @Override
  public int sampleWeighted(WeightFunction weights) {
    Int2DoubleMap weighted = new Int2DoubleOpenHashMap(map);
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      entry.setValue(weights.accept(entry.getIntKey(), entry.getDoubleValue()));
    }
    return Sample.sample(weighted);
  }

  @Override
  public DistributionBuilder map(IntUnaryOperator map) {
    Builder builder = new Builder();
    for (Int2DoubleMap.Entry entry : this.map.int2DoubleEntrySet()) {
      int key = map.applyAsInt(entry.getIntKey());
      if (key >= 0) {
        builder.add(key, entry.getDoubleValue());
      }
    }
    return builder;
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

    if (!support.equals(other.support())) {
      return false;
    }
    for (Int2DoubleMap.Entry entry : this.map.int2DoubleEntrySet()) {
      if (!Util.isEqual(entry.getDoubleValue(), other.get(entry.getIntKey()))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    if (lazyHash == 0) {
      lazyHash = support.hashCode();
    }
    return lazyHash;
  }

  @Override
  public String toString() {
    return map.toString();
  }

  @Override
  public void forEach(DistributionConsumer action) {
    for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
      action.accept(entry.getIntKey(), entry.getDoubleValue());
    }
  }

  public static class Builder extends AbstractBuilder {
    @Override
    public Distribution scaled() {
      NatBitSet support = support();
      if (support.isEmpty()) {
        return EmptyDistribution.INSTANCE;
      }
      int size = support.size();
      if (size == 1) {
        int key = support.firstInt();
        return new MapDistribution(key, 1.0d);
      }

      double sum = 0.0d;
      Int2DoubleMap map = map();
      DoubleIterator iterator = map.values().iterator();
      while (iterator.hasNext()) {
        sum += iterator.nextDouble();
      }
      for (Int2DoubleMap.Entry entry : map.int2DoubleEntrySet()) {
        entry.setValue(entry.getDoubleValue() / sum);
      }
      return new MapDistribution(map, support);
    }

    @Override
    public Distribution build() {
      NatBitSet support = support();
      if (support.isEmpty()) {
        return EmptyDistribution.INSTANCE;
      }
      int size = support.size();
      if (size == 1) {
        int key = support.firstInt();
        return new MapDistribution(key, 1.0d);
      }

      Int2DoubleMap map = map();
      return new MapDistribution(map, support);
    }
  }
}
