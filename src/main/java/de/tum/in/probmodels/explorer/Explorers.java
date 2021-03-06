package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.Model;

// Wrapper class to return whitebox/blackbox/greybox explorers
public class Explorers {
  private Explorers(){

  }

  public static <S, M extends Model> Explorer<S, M> getExplorer(M partialModel, Generator<S> generator,
                                                                InformationLevel informationLevel, boolean removeSelfLoops,
                                                                long timeout){
    if (informationLevel==InformationLevel.BLACKBOX){
      return new BlackExplorer<>(partialModel, generator, removeSelfLoops, timeout);
    }
    else if (informationLevel==InformationLevel.WHITEBOX){
      return DefaultExplorer.of(partialModel, generator, removeSelfLoops);
    }
    else {
      return new GreyExplorer<>(partialModel, generator, removeSelfLoops, timeout);
    }
  }
}
