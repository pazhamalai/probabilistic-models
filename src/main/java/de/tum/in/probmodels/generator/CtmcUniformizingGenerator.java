package de.tum.in.probmodels.generator;

import static com.google.common.base.Preconditions.checkArgument;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.Collection;
import java.util.Collections;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class CtmcUniformizingGenerator extends PrismGenerator {
  private final double rate;

  public CtmcUniformizingGenerator(ModelGenerator generator, double rate) {
    super(generator);
    this.rate = rate;
  }

  @Override
  public Collection<Choice<State>> getChoices(State state) throws PrismException {
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
        State target = entry.getKey();
        double probability = entry.getDoubleValue();
        checkArgument(probability <= rate,
            "Rate %s smaller than transition probability %s", rate, probability);

        if (!target.equals(state)) {
          double uniformizedProbability = probability / rate;
          sum += uniformizedProbability;
          double oldValue = map.put(target, uniformizedProbability);
          assert Double.isNaN(oldValue);
        }
      }
    }

    if (sum < 1.0d) {
      map.put(state, 1 - sum);
    }
    return Collections.singleton(Choice.of(map));
  }

  @Override
  public String toString() {
    return String.format("CTMCUniformizingGen(%.2f)", rate);
  }
}
