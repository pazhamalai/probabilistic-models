package de.tum.in.probmodels.util;

import static de.tum.in.probmodels.util.Util.isZero;

import it.unimi.dsi.fastutil.doubles.DoubleIterator;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.IntList;
import java.util.List;
import java.util.Random;

public final class Sample {
  private static final Random random = new Random();

  private Sample() {
    // Empty
  }

  public static int sample(int max) {
    return random.nextInt(max);
  }

  public static int sample(double[] values) {
    if (values.length == 0) {
      return -1;
    }
    if (values.length == 1) {
      return values[0] == 0.0d ? -1 : 0;
    }

    double sum = 0.0d;
    for (double value : values) {
      sum += value;
    }

    if (isZero(sum)) {
      return -1;
    }

    // Sample a random value in [0, sum)
    double sampledValue = random.nextDouble() * sum;
    // Search the successor corresponding to this value
    double partialSum = 0.0d;
    for (int i = 0; i < values.length; i++) {
      partialSum += values[i];
      if (partialSum >= sampledValue) {
        return i;
      }
    }

    throw new AssertionError("Not sampling any value");
  }

  public static int sample(Int2DoubleMap distribution) {
    if (distribution.isEmpty()) {
      return -1;
    }
    if (distribution.size() == 1) {
      return distribution.keySet().iterator().nextInt();
    }

    double sum = 0.0d;
    DoubleIterator iterator = distribution.values().iterator();
    while (iterator.hasNext()) {
      sum += iterator.nextDouble();
    }

    // Sample a random value in [0, sum)
    double sampledValue = random.nextDouble() * sum;
    // Search the successor corresponding to this value
    double partialSum = 0.0d;
    for (Int2DoubleMap.Entry entry : distribution.int2DoubleEntrySet()) {
      partialSum += entry.getDoubleValue();
      if (partialSum >= sampledValue) {
        return entry.getIntKey();
      }
    }

    throw new AssertionError("Not sampling any value");
  }

  public static int sampleUniform(IntList values) {
    int size = values.size();
    if (size == 0) {
      return -1;
    }
    if (size == 1) {
      return values.getInt(0);
    }
    return values.getInt(random.nextInt(size));
  }

  public static int sampleUniform(int[] values, int max) {
    assert max <= values.length;
    if (max == 0) {
      return -1;
    }
    if (max == 1) {
      return values[0];
    }
    return values[random.nextInt(max)];
  }

  public static <T> T sampleUniform(List<? extends T> values) {
    int size = values.size();
    if (size == 0) {
      return null;
    }
    return size == 1 ? values.get(0) : values.get(random.nextInt(size));
  }
}
