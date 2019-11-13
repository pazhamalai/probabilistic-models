package de.tum.in.probmodels.generator;

import de.tum.in.probmodels.util.annotation.Tuple;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMaps;
import javax.annotation.Nullable;
import org.immutables.value.Value;

@Value.Immutable
@Tuple
public
interface Choice<S> {
  @Nullable
  Object label();

  Object2DoubleMap<S> transitions();


  static <S> Choice<S> of(Object2DoubleMap<S> transitions) {
    return of(null, transitions);
  }

  static <S> Choice<S> of(@Nullable Object label, Object2DoubleMap<S> transitions) {
    return ChoiceTuple.create(label, transitions);
  }

  static <S> Choice<S> of(@Nullable Object label, S target) {
    return ChoiceTuple.create(label, Object2DoubleMaps.singleton(target, 1.0d));
  }

  static <S> Choice<S> selfLoop(S target) {
    return of(null, Object2DoubleMaps.singleton(target, 1.0d));
  }
}
