package de.tum.in.probmodels.model;

import de.tum.in.probmodels.util.annotation.HashedTuple;
import javax.annotation.Nullable;
import org.immutables.value.Value;

@Value.Immutable
@HashedTuple
public abstract class Action {
  public abstract Distribution distribution();

  @Nullable
  public abstract Object label();


  public static Action of(Distribution distribution) {
    return ActionTuple.create(distribution, null);
  }

  public static Action of(Distribution distribution, Object label) {
    return ActionTuple.create(distribution, label);
  }
}
