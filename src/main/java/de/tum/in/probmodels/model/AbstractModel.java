package de.tum.in.probmodels.model;

import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import explicit.PredecessorRelation;
import explicit.StateValues;
import java.io.File;
import java.util.BitSet;
import java.util.List;
import java.util.Set;
import parser.State;
import parser.Values;
import parser.VarList;
import prism.PrismComponent;
import prism.PrismException;
import prism.PrismLog;

public abstract class AbstractModel implements Model {
  @Override
  public int getNumChoices(int state) {
    return getChoices(state).size();
  }

  @Override
  public void buildFromPrismExplicit(String filename) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void setStatesList(List<State> statesList) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void addLabel(String name, BitSet states) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getNumInitialStates() {
    return Iterators.size(getInitialStates().iterator());
  }

  @Override
  public int getFirstInitialState() {
    return getInitialStates().iterator().nextInt();
  }

  @Override
  public boolean isInitialState(int i) {
    return Iterables.contains(getInitialStates(), i);
  }

  @Override
  public int getNumDeadlockStates() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterable<Integer> getDeadlockStates() {
    throw new UnsupportedOperationException();
  }

  @Override
  public StateValues getDeadlockStatesList() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getFirstDeadlockState() {
    return getDeadlockStates().iterator().next();
  }

  @Override
  public boolean isDeadlockState(int i) {
    return Iterables.contains(getDeadlockStates(), i);
  }

  @Override
  public List<State> getStatesList() {
    throw new UnsupportedOperationException();
  }

  @Override
  public VarList getVarList() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Values getConstantValues() {
    throw new UnsupportedOperationException();
  }

  @Override
  public BitSet getLabelStates(String name) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Set<String> getLabels() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean hasLabel(String name) {
    throw new UnsupportedOperationException();
  }


  @Override
  public void findDeadlocks(boolean fix) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void checkForDeadlocks() throws PrismException {
    checkForDeadlocks(null);
  }

  @Override
  public void checkForDeadlocks(BitSet except) throws PrismException {
    for (int i = 0; i < getNumStates(); i++) {
      if ((except == null || !except.get(i)) && getChoices(i).isEmpty()) {
        throw new PrismException("Deadlock in state " + i);
      }
    }
  }

  @Override
  public void exportToPrismExplicit(String baseFilename) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportToPrismExplicitTra(String filename) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportToPrismExplicitTra(File file) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportToPrismExplicitTra(PrismLog log) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportToPrismLanguage(String filename) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportStates(int exportType, VarList varList, PrismLog log) {
    throw new UnsupportedOperationException();
  }

  @Override
  public String infoString() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String infoStringTable() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean hasStoredPredecessorRelation() {
    return false;
  }

  @Override
  public PredecessorRelation getPredecessorRelation(PrismComponent parent, boolean storeIfNew) {
    return new PredecessorRelation(this);
  }

  @Override
  @SuppressWarnings("PMD.EmptyMethodInAbstractClassShouldBeAbstract")
  public void clearPredecessorRelation() {
    // Intentionally empty
  }
}
