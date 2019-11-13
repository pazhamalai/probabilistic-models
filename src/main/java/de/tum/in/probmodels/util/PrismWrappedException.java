package de.tum.in.probmodels.util;

import prism.PrismException;

public class PrismWrappedException extends RuntimeException {
  public PrismWrappedException(PrismException e) {
    super(e);
  }
}
