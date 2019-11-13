package de.tum.in.probmodels.util;

import java.util.function.Predicate;
import parser.State;
import parser.ast.Expression;
import prism.PrismLangException;

public class PrismExpressionWrapper implements Predicate<State> {
  private final Expression expression;

  public PrismExpressionWrapper(Expression expression) {
    this.expression = expression;
  }

  @Override
  public boolean test(State state) {
    try {
      return expression.evaluateBoolean(state);
    } catch (PrismLangException e) {
      throw new PrismWrappedException(e);
    }
  }

  @Override
  public String toString() {
    return expression.toString();
  }
}
