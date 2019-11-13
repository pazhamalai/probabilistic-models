package de.tum.in.probmodels.util;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import javax.annotation.Nullable;

public interface Result<S, R> {
  Collection<S> states();

  @Nullable
  R get(S state);


  static <S, R> Result<S, R> of(Map<S, R> results) {
    return new MapResult<>(results);
  }

  static <S, R> Result<S, R> of(S state, R result) {
    return new SingletonResult<>(state, result);
  }


  final class MapResult<S, R> implements Result<S, R> {
    private final Map<S, R> results;

    private MapResult(Map<S, R> results) {
      this.results = Map.copyOf(results);
    }

    @Override
    public Collection<S> states() {
      return results.keySet();
    }

    @Override
    public R get(S state) {
      return results.get(state);
    }
  }

  final class SingletonResult<S, R> implements Result<S, R> {
    private final S state;
    private final R result;

    public SingletonResult(S state, R result) {
      this.state = state;
      this.result = result;
    }

    @Override
    public Collection<S> states() {
      return Collections.singleton(state);
    }

    @Override
    public R get(S state) {
      return this.state.equals(state) ? result : null;
    }
  }
}
