package de.tum.in.probmodels.model.distribution;

public interface DistributionBuilder {
  void add(int j, double prob);

  void set(int j, double prob);

  Distribution scaled();

  Distribution build();

  boolean isEmpty();
}
