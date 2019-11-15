package de.tum.in.probmodels.generator;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class MdpGenerator extends PrismGenerator {
  public MdpGenerator(ModelGenerator generator) {
    super(generator);
  }

  @Override
  public Collection<Choice<State>> getChoices(State state) throws PrismException {
    ModelGenerator generator = generator();
    generator.exploreState(state);

    int choiceCount = generator.getNumChoices();
    List<Choice<State>> choices = new ArrayList<>(choiceCount);

    for (int choiceIndex = 0; choiceIndex < choiceCount; choiceIndex++) {
      int transitionCount = generator.getNumTransitions(choiceIndex);
      Object actionLabel = generator.getChoiceAction(choiceIndex);

      Object2DoubleMap<State> map = new Object2DoubleOpenHashMap<>(transitionCount);
      map.defaultReturnValue(Double.NaN);
      transitions(choiceIndex, transitionCount).forEach(entry -> {
        double oldValue = map.put(entry.getKey(), entry.getDoubleValue());
        assert Double.isNaN(oldValue);
      });
      choices.add(Choice.of(actionLabel, map));
    }

    return choices;
  }
}
