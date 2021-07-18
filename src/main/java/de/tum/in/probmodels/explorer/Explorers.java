package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.Model;

public class Explorers {
  private Explorers(){

  }

  public static <S, M extends Model> Explorer<S, M> getExplorer(M partialModel, Generator<S> generator,
                                                                InformationLevel informationLevel, boolean removeSelfLoops){
    if (informationLevel==InformationLevel.BLACKBOX){
      return BlackExplorer.of(partialModel, generator, removeSelfLoops);
    }
    else if (informationLevel==InformationLevel.WHITEBOX){
      return DefaultExplorer.of(partialModel, generator, removeSelfLoops);
    }
    else {
      throw new UnsupportedOperationException("Only models with Blackbox and Whitebox information levels are supported at the moment.");
    }
  }
}
