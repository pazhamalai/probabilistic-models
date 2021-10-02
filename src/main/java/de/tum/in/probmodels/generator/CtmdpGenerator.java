package de.tum.in.probmodels.generator;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class CtmdpGenerator extends PrismGenerator {

    public CtmdpGenerator(ModelGenerator generator) {
        super(generator);
    }

    @Override
    protected Collection<Choice<State>> getChoices(State state) throws PrismException {
        ModelGenerator generator = generator();
        generator.exploreState(state);

        int choiceCount = generator.getNumChoices();
        List<Choice<State>> choices = new ArrayList<>(choiceCount);

        for (int choiceIndex = 0; choiceIndex < choiceCount; choiceIndex++) {
            int transitionCount = generator.getNumTransitions(choiceIndex);
            Object actionLabel = generator.getChoiceAction(choiceIndex);

            Object2DoubleMap<State> map = new Object2DoubleOpenHashMap<>(transitionCount);
            map.defaultReturnValue(Double.NaN);
            transitions(choiceIndex, transitionCount)
                    .forEach(entry -> map.mergeDouble(entry.getKey(), entry.getDoubleValue(), Double::sum));
            choices.add(Choice.of(actionLabel, map));
        }

        return choices;
    }
}
