package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.Distribution;
import de.tum.in.probmodels.model.DistributionBuilder;
import de.tum.in.probmodels.model.Distributions;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntSet;

import java.util.function.IntConsumer;

public class UniformizedMecBuilder {
    private final NatBitSet states;
    private final Int2ObjectMap<IntSet> actions;
    private final Int2ObjectMap<Int2ObjectMap<DistributionBuilder>> distributionBuilders;

    public UniformizedMecBuilder(NatBitSet states, Int2ObjectMap<IntSet> actions) {
        this.states = states;
        this.actions = actions;
        this.distributionBuilders = new Int2ObjectOpenHashMap<>();

        initDistributionBuilders();
    }

    public void addUniformizedTransition(int state, int action, int successor, double uniformizedRate) {
        distributionBuilders.get(state).get(action).add(successor, uniformizedRate);
    }

    public UniformizedMEC build() {
        Int2ObjectMap<Int2ObjectMap<Distribution>> distributions = new Int2ObjectOpenHashMap<>();

        for (int state : states) {
            Int2ObjectMap<Distribution> stateDistributions = new Int2ObjectOpenHashMap<>();
            for (int action : actions.get(state)) {
                DistributionBuilder builder = distributionBuilders.get(state).get(action);
                stateDistributions.put(action, builder.build());
            }
            distributions.put(state, stateDistributions);
        }

        return new UniformizedMEC(states, actions, distributions);
    }

    private void initDistributionBuilders() {
        states.forEach((IntConsumer) this::initStateDistributionBuilders);
    }

    private void initStateDistributionBuilders(int state) {
        Int2ObjectMap<DistributionBuilder> stateDistributionBuilder = new Int2ObjectOpenHashMap<>();

        for (Integer action : actions.get(state)) {
            stateDistributionBuilder.put(action.intValue(), Distributions.defaultBuilder());
        }

        distributionBuilders.put(state, stateDistributionBuilder);
    }
}
