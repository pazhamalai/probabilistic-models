package de.tum.in.probmodels.util;

import static de.tum.in.probmodels.util.Util.isZero;

import de.tum.in.probmodels.model.Distribution;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.IntList;
import java.util.List;
import java.util.Random;

public final class Sample {
  private static final Random random = new Random();

  private Sample() {
    // Empty
  }

  public static int sample(Distribution distribution) {
    if (distribution.isEmpty()) {
      return -1;
    }
    if (distribution.size() == 1) {
      return distribution.support().firstInt();
    }

    int[] keys = new int[distribution.size()];
    double[] values = new double[distribution.size()];

    int index = 0;
    for (Int2DoubleMap.Entry entry : distribution) {
      keys[index] = entry.getIntKey();
      values[index] = entry.getDoubleValue();
      index += 1;
    }

    int sample = sample(values, values.length);
    return sample == -1 ? -1 : keys[sample];
  }

  public static int sample(double[] values, int bound) {
    double sum = 0.0d;
    for (int i = 0; i < bound; i++) {
      sum += values[i];
    }

    if (isZero(sum)) {
      return -1;
    }

    // Sample a random value in [0, sum)
    double sampledValue = random.nextDouble() * sum;
    // Search the successor corresponding to this value
    double partialSum = 0.0d;
    for (int i = 0; i < bound; i++) {
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

    int[] keys = new int[distribution.size()];
    double[] values = new double[distribution.size()];

    int index = 0;
    for (Int2DoubleMap.Entry entry : distribution.int2DoubleEntrySet()) {
      keys[index] = entry.getIntKey();
      values[index] = entry.getDoubleValue();
      index += 1;
    }
    int sample = sample(values, values.length);
    return sample == -1 ? -1 : keys[sample];
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
