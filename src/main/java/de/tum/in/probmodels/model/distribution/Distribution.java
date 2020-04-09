package de.tum.in.probmodels.model.distribution;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import java.util.BitSet;
import java.util.Iterator;
import java.util.Map;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntUnaryOperator;

public interface Distribution extends Iterable<Int2DoubleMap.Entry> {
  @FunctionalInterface
  interface DistributionConsumer {
    void accept(int state, double probability);
  }

  @FunctionalInterface
  interface WeightFunction {
    double accept(int state, double probability);
  }

  double get(int j);


  NatBitSet support();

  default boolean contains(int j) {
    return support().contains(j);
  }

  default boolean isEmpty() {
    return support().isEmpty();
  }

  default int size() {
    return support().size();
  }


  double sum();

  double sumWeighted(IntToDoubleFunction f);

  double sumWeightedExceptJacobi(IntToDoubleFunction f, int state);


  int sample();

  int sampleWeighted(WeightFunction weights);

  DistributionBuilder map(IntUnaryOperator map);

  void forEach(DistributionConsumer action);

  @Deprecated
  Iterator<Map.Entry<Integer, Double>> objectIterator();


  default boolean isSubsetOf(BitSet set) {
    if (size() > set.size()) {
      return false;
    }
    return NatBitSets.asSet(set).containsAll(support());
  }

  default boolean containsOneOf(BitSet set) {
    if (set.isEmpty()) {
      return true;
    }
    return support().intersects(NatBitSets.asSet(set));
  }

  default boolean containsOneOf(NatBitSet set) {
    if (set.isEmpty()) {
      return true;
    }
    return support().intersects(set);
  }
}
