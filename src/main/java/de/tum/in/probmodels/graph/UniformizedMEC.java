package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.Distribution;
import it.unimi.dsi.fastutil.ints.*;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

// This class holds info about mec states, actions and the uniformized rates of every transition.
// TODO add action label information
// TODO OR STORE IT AS ACTION OBJECT ITSELF
// TODO Represent it as MDP object
public class UniformizedMEC {
    private final NatBitSet states;
    private final Int2ObjectMap<IntSet> actions;
    private final Int2ObjectMap<Int2ObjectMap<Distribution>> uniformizedDistribution;

    public UniformizedMEC(NatBitSet states, Int2ObjectMap<IntSet> actions, Int2ObjectMap<Int2ObjectMap<Distribution>> uniformizedDistribution) {
        this.states = states;
        this.actions = actions;
        this.uniformizedDistribution = uniformizedDistribution;
    }

    public Set<Integer> getStates() {
        return Collections.unmodifiableSet(states);
    }

    public Map<Integer, IntSet> getActions() {
        return Collections.unmodifiableMap(actions);
    }

    public Distribution getUniformizedDistribution(int state, int action) {
        assert states.contains(state);
        assert actions.get(state).contains(action);

        return uniformizedDistribution.get(state).get(action);
    }
}
