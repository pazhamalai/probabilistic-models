package de.tum.in.probmodels.generator;

import de.tum.in.probmodels.util.annotation.HashedTuple;
import javax.annotation.Nullable;
import org.immutables.value.Value;

@Value.Immutable
@HashedTuple
public abstract class ProductState<S, A> {
  // TODO Nicer than NULL
  @Nullable
  public abstract S system();

  public abstract A automaton();


  public static <S, A> ProductState<S, A> of(@Nullable S system, A automaton) {
    return ProductStateTuple.create(system, automaton);
  }


  @Override
  public String toString() {
    return String.format("[%s]x[%s]", system(), automaton());
  }
}
