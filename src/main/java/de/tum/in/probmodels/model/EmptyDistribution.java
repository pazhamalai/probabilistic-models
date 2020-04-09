package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleMaps;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntUnaryOperator;

public final class EmptyDistribution implements Distribution {
  static final EmptyDistribution INSTANCE = new EmptyDistribution();

  private EmptyDistribution() {
    // empty
  }

  @Override
  public double get(int j) {
    return 0.0d;
  }

  @Override
  public NatBitSet support() {
    return NatBitSets.emptySet();
  }

  @Override
  public double sum() {
    return 0.0d;
  }

  @Override
  public double sumWeighted(IntToDoubleFunction f) {
    return 0.0d;
  }

  @Override
  public double sumWeightedExceptJacobi(IntToDoubleFunction f, int state) {
    return 0.0d;
  }

  @Override
  public int sample() {
    return -1;
  }

  @Override
  public int sampleWeighted(WeightFunction weights) {
    return -1;
  }

  @Override
  public DistributionBuilder map(IntUnaryOperator map) {
    return Distributions.defaultBuilder();
  }

  @Override
  public void forEach(DistributionConsumer action) {
    // empty
  }


  @Override
  public Iterator<Map.Entry<Integer, Double>> objectIterator() {
    return Collections.emptyIterator();
  }

  @Override
  public Iterator<Int2DoubleMap.Entry> iterator() {
    return Int2DoubleMaps.EMPTY_MAP.int2DoubleEntrySet().iterator();
  }

  @Override
  public boolean equals(Object obj) {
    return obj == this;
  }

  @Override
  public int hashCode() {
    return getClass().hashCode();
  }

  @Override
  public String toString() {
    return "{}";
  }
}
