package de.tum.in.probmodels.graph;

import de.tum.in.probmodels.model.Distribution;
import it.unimi.dsi.fastutil.ints.Int2DoubleFunction;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectFunction;

public class MecUniformizer {

    // Given a state and an action, rateFunction should return the rate associated with the pair.
    private Int2ObjectFunction<Int2DoubleFunction> rateFunction;

    // Given a state-action, return a distribution
    private Int2ObjectFunction<Int2ObjectFunction<Distribution>> distributionFunction;

    public void setRateFunction(Int2ObjectFunction<Int2DoubleFunction> rateFunction) {
        this.rateFunction = rateFunction;
    }

    public void setDistributionFunction(Int2ObjectFunction<Int2ObjectFunction<Distribution>> distributionFunction) {
        this.distributionFunction = distributionFunction;
    }

    public UniformizedMEC uniformize(Mec mec, double maxRate) {
        UniformizedMecBuilder uniformizedMecBuilder = new UniformizedMecBuilder(mec.states, mec.actions);

        // For every state-action pair in MEC,
        // get the uniformized rate, set self loop if needed, adjust the transition probabilities
        for (Integer state : mec.states) {
            for (Integer action : mec.actions.get(state.intValue())) {
                uniformizeStateAction(state, action, uniformizedMecBuilder, maxRate);
            }
        }

        return uniformizedMecBuilder.build();
    }

    private void uniformizeStateAction(int state, int action, UniformizedMecBuilder uniformizedMecBuilder, double maxRate) {
        double rate = rateFunction.apply(state).apply(action);
        double rateToMaxRate = rate/maxRate;

        boolean hasSelfLoop = false;
        Distribution distribution = distributionFunction.apply(state).apply(action);
        for (Int2DoubleMap.Entry entry : distribution) {
            int successor = entry.getIntKey();
            double prob = entry.getDoubleValue();

            if (successor == state) {
                hasSelfLoop = true;
            }

            double uniformizedRate = computeUniformizedRate(rateToMaxRate, prob, successor == state);
            uniformizedMecBuilder.addUniformizedTransition(state, action, successor, uniformizedRate);
        }

        if (!hasSelfLoop) {
            uniformizedMecBuilder.addUniformizedTransition(state, action, state, 1-rateToMaxRate);
        }
    }

    private double computeUniformizedRate(double rateToMaxRate, double transitionProbability, boolean isSelfLoop) {
        double uniformizedRate;

        if (isSelfLoop) {
            uniformizedRate = (rateToMaxRate * transitionProbability) +  (1 - rateToMaxRate);
        }
        else {
            uniformizedRate = rateToMaxRate * transitionProbability;
        }

        return uniformizedRate;
    }
}
