package de.tum.in.probmodels.generator;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.Collection;
import java.util.Collections;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class DtmcGenerator extends PrismGenerator {
  public DtmcGenerator(ModelGenerator generator) {
    super(generator);
  }

  @Override
  public Collection<Choice<State>> getChoices(State state) throws PrismException {
    ModelGenerator generator = generator();
    generator.exploreState(state);

    int choiceCount = generator.getNumChoices();
    assert choiceCount <= 1;

    int transitionCount = generator.getNumTransitions();
    Object2DoubleMap<State> map = new Object2DoubleOpenHashMap<>(transitionCount);
    map.defaultReturnValue(Double.NaN);
    transitions(0, transitionCount).forEach(entry -> {
      double oldValue = map.put(entry.getKey(), entry.getDoubleValue());
      assert Double.isNaN(oldValue);
    });
    return Collections.singleton(Choice.of(map));
  }

  @Override
  public String toString() {
    return "DTMCgen";
  }
}
