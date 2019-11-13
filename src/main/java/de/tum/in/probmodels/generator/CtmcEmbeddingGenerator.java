package de.tum.in.probmodels.generator;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.Collection;
import java.util.Collections;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class CtmcEmbeddingGenerator extends PrismGenerator {
  public CtmcEmbeddingGenerator(ModelGenerator generator) {
    super(generator);
  }

  @Override
  public Collection<Choice<State>> choices(State state) throws PrismException {
    ModelGenerator generator = generator();
    generator.exploreState(state);

    int choiceCount = generator.getNumChoices();
    int transitionCount = generator.getNumTransitions();

    Object2DoubleMap<State> map = new Object2DoubleOpenHashMap<>(transitionCount);
    map.defaultReturnValue(Double.NaN);

    double sum = 0.0d;
    for (int choiceIndex = 0; choiceIndex < choiceCount; choiceIndex++) {
      int choiceTransitionCount = generator.getNumTransitions(choiceIndex);

      for (Object2DoubleMap.Entry<State> entry : transitions(choiceIndex, choiceTransitionCount)) {
        double probability = entry.getDoubleValue();

        double oldValue = map.put(entry.getKey(), probability);
        assert Double.isNaN(oldValue);

        sum += probability;
      }
    }

    if (map.isEmpty()) {
      return Collections.singleton(Choice.selfLoop(state));
    }
    for (Object2DoubleMap.Entry<State> entry : map.object2DoubleEntrySet()) {
      entry.setValue(entry.getDoubleValue() / sum);
    }
    return Collections.singleton(Choice.of(map));
  }
}
