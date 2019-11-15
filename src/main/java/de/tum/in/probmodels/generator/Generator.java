package de.tum.in.probmodels.generator;

import java.util.Collection;

public interface Generator<S> {
  Collection<S> initialStates();

  Collection<Choice<S>> choices(S state);
}
