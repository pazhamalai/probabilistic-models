package de.tum.in.probmodels.model;

import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;
import java.util.Collection;

public abstract class DefaultModel extends AbstractModel {
  private final IntList initialStates = new IntArrayList();
  private int numStates = 0;

  @Override
  public int addState() {
    numStates += 1;
    return numStates - 1;
  }

  @Override
  public void addStates(int numToAdd) {
    numStates += numToAdd;
  }

  @Override
  public int getNumStates() {
    return numStates;
  }

  @Override
  public void addInitialState(int i) {
    initialStates.add(i);
  }

  @Override
  public void setInitialStates(Collection<Integer> initialStates) {
    this.initialStates.clear();
    this.initialStates.addAll(initialStates);
  }

  @Override
  public IntList getInitialStates() {
    return initialStates;
  }
}
