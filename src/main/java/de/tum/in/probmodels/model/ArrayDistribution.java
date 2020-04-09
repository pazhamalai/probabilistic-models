package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.util.Sample;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.ints.AbstractInt2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.IntIterator;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

public class ArrayDistribution implements Distribution {
  private final int[] successors;
  private final double[] probabilities;
  private final NatBitSet support;
  private int lazyHash = 0;

  ArrayDistribution(int key, double probability) {
    successors = new int[] {key};
    probabilities = new double[] {probability};
    support = NatBitSets.singleton(key);
  }

  @SuppressWarnings({"PMD.ArrayIsStoredDirectly", "AssignmentOrReturnOfFieldWithMutableType"})
  ArrayDistribution(int[] successors, double[] probabilities, NatBitSet support) {
    assert IntStream.range(0, successors.length - 1)
        .allMatch(i -> successors[i] < successors[i + 1]);
    assert Arrays.stream(successors).allMatch(support::contains)
        && successors.length == support.size();

    this.successors = successors;
    this.probabilities = probabilities;
    this.support = support;
  }

  @Override
  public double get(int key) {
    int index = Arrays.binarySearch(successors, key);
    return index >= 0 ? probabilities[index] : 0.0d;
  }

  @Override
  public boolean contains(int j) {
    return support.contains(j);
  }

  @Override
  public NatBitSet support() {
    return support;
  }

  @Override
  public boolean isEmpty() {
    return successors.length == 0;
  }

  @Override
  public int size() {
    return successors.length;
  }

  @Override
  public double sum() {
    double d = 0.0;
    for (double value : probabilities) {
      d += value;
    }
    return d;
  }

  @Override
  public double sumWeighted(IntToDoubleFunction f) {
    double d = 0.0;
    for (int i = 0; i < successors.length; i++) {
      d += f.applyAsDouble(successors[i]) * probabilities[i];
    }
    return d;
  }

  @Override
  public double sumWeightedExceptJacobi(IntToDoubleFunction f, int state) {
    double sum = 0.0d;
    double weight = 0.0d;
    for (int i = 0; i < successors.length; i++) {
      int s = successors[i];
      if (s != state) {
        double probability = probabilities[i];
        sum += f.applyAsDouble(s) * probability;
        weight += probability;
      }
    }
    return weight == 0.0d ? 0.0d : sum / weight;
  }


  @Override
  public int sample() {
    return successors[Sample.sample(probabilities)];
  }

  @Override
  public int sampleWeighted(WeightFunction weights) {
    double[] weightArray = new double[successors.length];
    for (int i = 0; i < successors.length; i++) {
      weightArray[i] = weights.accept(successors[i], probabilities[i]);
    }
    int sample = Sample.sample(weightArray);
    return sample == -1 ? -1 : successors[sample];
  }


  @Override
  public DistributionBuilder map(IntUnaryOperator map) {
    Builder builder = new Builder();
    for (int i = 0; i < successors.length; i++) {
      int key = map.applyAsInt(successors[i]);
      if (key >= 0) {
        builder.add(key, probabilities[i]);
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

    if (other instanceof ArrayDistribution) {
      ArrayDistribution array = (ArrayDistribution) other;
      assert Arrays.equals(successors, array.successors);
      for (int i = 0; i < successors.length; i++) {
        if (!Util.isEqual(probabilities[i], array.probabilities[i])) {
          return false;
        }
      }
      return true;
    }

    for (int i = 0; i < successors.length; i++) {
      if (!Util.isEqual(probabilities[i], other.get(successors[i]))) {
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
    StringBuilder builder = new StringBuilder("{");
    for (int i = 0; i < successors.length; i++) {
      builder.append(successors[i]).append(": ").append(probabilities[i]);
      if (i < successors.length - 1) {
        builder.append(", ");
      }
    }
    builder.append('}');
    return builder.toString();
  }

  @Override
  public void forEach(DistributionConsumer action) {
    for (int i = 0; i < successors.length; i++) {
      action.accept(successors[i], probabilities[i]);
    }
  }

  @Override
  public Iterator<Map.Entry<Integer, Double>> objectIterator() {
    return new EntryObjectIterator(successors, probabilities);
  }

  @Override
  public Iterator<Int2DoubleMap.Entry> iterator() {
    return new EntryIterator(successors, probabilities);
  }

  private static class EntryIterator implements Iterator<Int2DoubleMap.Entry> {
    private final int[] keys;
    private final double[] probabilities;
    private int index = 0;

    @SuppressWarnings("PMD.ArrayIsStoredDirectly")
    EntryIterator(int[] keys, double[] probabilities) {
      this.keys = keys;
      this.probabilities = probabilities;
    }

    @Override
    public boolean hasNext() {
      return index < keys.length;
    }

    @Override
    public Int2DoubleMap.Entry next() {
      if (index == keys.length) {
        throw new NoSuchElementException();
      }
      var entry = new AbstractInt2DoubleMap.BasicEntry(keys[index], probabilities[index]);
      index += 1;
      return entry;
    }
  }


  private static class EntryObjectIterator implements Iterator<Map.Entry<Integer, Double>> {
    private final int[] keys;
    private final double[] probabilities;
    private int index = 0;

    @SuppressWarnings("PMD.ArrayIsStoredDirectly")
    EntryObjectIterator(int[] keys, double[] probabilities) {
      this.keys = keys;
      this.probabilities = probabilities;
    }

    @Override
    public boolean hasNext() {
      return index < keys.length;
    }

    @Override
    public Map.Entry<Integer, Double> next() {
      if (index == keys.length) {
        throw new NoSuchElementException();
      }
      var entry = new AbstractInt2DoubleMap.BasicEntry(keys[index], probabilities[index]);
      index += 1;
      return entry;
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
        return new ArrayDistribution(key, 1.0d);
      }

      int[] keys = new int[size];
      double[] probabilities = new double[size];

      int index = 0;
      double sum = 0.0d;
      Int2DoubleMap map = map();
      IntIterator keyIterator = support.iterator();
      while (keyIterator.hasNext()) {
        int key = keyIterator.nextInt();
        double probability = map.get(key);
        keys[index] = key;
        probabilities[index] = probability;
        index += 1;
        sum += probability;
      }
      for (int i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sum;
      }
      return new ArrayDistribution(keys, probabilities, support);
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
        return new ArrayDistribution(key, 1.0d);
      }

      Int2DoubleMap map = map();
      int[] keys = new int[size];
      double[] probabilities = new double[size];

      int index = 0;
      IntIterator keyIterator = support.iterator();
      while (keyIterator.hasNext()) {
        int key = keyIterator.nextInt();
        double probability = map.get(key);
        keys[index] = key;
        probabilities[index] = probability;
        index += 1;
      }
      return new ArrayDistribution(keys, probabilities, support);
    }
  }
}
