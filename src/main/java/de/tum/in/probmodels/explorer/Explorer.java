package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.model.Action;
import de.tum.in.probmodels.model.Distribution;
import explicit.Model;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.List;
import prism.PrismException;

public interface Explorer<S, M extends Model> {
  IntCollection initialStates();

  S exploreState(int stateId) throws PrismException;

  boolean isExploredState(int stateId);

  List<Action> getActions(int stateId);

  List<Distribution> getChoices(int stateId);


  int exploredStateCount();

  IntSet exploredStates();

  M model();

  S getState(int stateId);

  int getStateId(S state);
}
