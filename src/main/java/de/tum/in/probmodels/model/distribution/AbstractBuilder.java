package de.tum.in.probmodels.model.distribution;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

public abstract class AbstractBuilder implements DistributionBuilder {
  private final Int2DoubleMap map = new Int2DoubleOpenHashMap();
  private final NatBitSet support = NatBitSets.set();

  @Override
  public void add(int j, double prob) {
    double old = map.getOrDefault(j, Double.NaN);
    if (Double.isNaN(old)) {
      map.put(j, prob);
      support.set(j);
    } else {
      double newValue = prob + old;
      if (newValue <= 0.0d) {
        support.clear(j);
        map.remove(j);
      } else {
        map.put(j, newValue);
      }
    }
  }

  @Override
  public void set(int j, double prob) {
    if (prob == 0.0d) {
      map.remove(j);
      support.clear(j);
    } else {
      map.put(j, prob);
      support.set(j);
    }
  }

  @SuppressWarnings("AssignmentOrReturnOfFieldWithMutableType")
  protected NatBitSet support() {
    return support;
  }

  @SuppressWarnings("AssignmentOrReturnOfFieldWithMutableType")
  protected Int2DoubleMap map() {
    return map;
  }

  @Override
  public boolean isEmpty() {
    return support.isEmpty();
  }
}
