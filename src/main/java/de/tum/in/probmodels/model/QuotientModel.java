package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.util.annotation.Tuple;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import org.immutables.value.Value;

@Value.Immutable
@Tuple
public abstract class QuotientModel<T extends Model> {
  public abstract T model();

  public abstract IntFunction<NatBitSet> quotientToStates();

  public abstract IntUnaryOperator stateToQuotient();
}
