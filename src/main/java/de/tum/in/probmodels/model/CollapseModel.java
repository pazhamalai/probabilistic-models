package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.List;
import prism.ModelType;

public interface CollapseModel<M extends Model> extends Model {
  IntList collapse(List<? extends IntSet> stateList);

  int representative(int state);

  boolean isRemoved(int state);

  NatBitSet removedStates();


  M getModel();

  @Override
  default ModelType getModelType() {
    return getModel().getModelType();
  }
}
