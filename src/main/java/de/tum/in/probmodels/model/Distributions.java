package de.tum.in.probmodels.model;

public final class Distributions {
  private Distributions() {
    // empty
  }

  public static DistributionBuilder defaultBuilder() {
    return new ArrayDistribution.Builder();
  }

  public static Distribution singleton(int key, double value) {
    DistributionBuilder builder = defaultBuilder();
    builder.add(key, value);
    return builder.build();
  }
}
