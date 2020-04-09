package de.tum.in.probmodels.util;

import de.tum.in.probmodels.model.distribution.MapDistribution;
import de.tum.in.probmodels.util.annotation.Tuple;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import org.immutables.value.Value;
import parser.PrismParser;
import parser.ast.Expression;
import parser.ast.LabelList;
import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import prism.PrismException;
import prism.UndefinedConstants;

public final class PrismHelper {
  private static final Logger logger = Logger.getLogger(PrismHelper.class.getName());

  private PrismHelper() {
    // Empty
  }

  public static PrismParseResult parse(String modelPath, @Nullable String propertiesPath,
      @Nullable String constantsString) throws PrismException, IOException {
    PrismParser parser = new PrismParser();

    logger.log(Level.FINE, "Reading model from {0}", modelPath);
    ModulesFile modulesFile;
    try (InputStream in = new BufferedInputStream(Files.newInputStream(Paths.get(modelPath)))) {
      modulesFile = parser.parseModulesFile(in, null);
    }
    modulesFile.tidyUp();

    @Nullable
    PropertiesFile propertiesFile;
    if (propertiesPath == null) {
      propertiesFile = null;
    } else {
      logger.log(Level.FINE, "Reading properties from {0}", propertiesPath);
      try (InputStream in = new BufferedInputStream(
          Files.newInputStream(Paths.get(propertiesPath)))) {
        propertiesFile = parser.parsePropertiesFile(modulesFile, in);
      }
      propertiesFile.tidyUp();
    }

    UndefinedConstants constants = new UndefinedConstants(modulesFile, propertiesFile);
    if (constantsString == null) {
      constants.checkAllDefined();
    } else {
      constants.defineUsingConstSwitch(constantsString);
    }

    modulesFile.setSomeUndefinedConstants(constants.getMFConstantValues());
    constants.iterateModel();

    List<Expression> expressionList = new ArrayList<>();

    if (propertiesFile != null) {
      propertiesFile.setSomeUndefinedConstants(constants.getPFConstantValues());

      for (int i = 0; i < propertiesFile.getNumProperties(); i++) {
        Expression property = propertiesFile.getProperty(i).deepCopy();
        // Combine label lists from model/property file, then expand property refs/labels in
        // property
        LabelList combinedLabelList = propertiesFile.getCombinedLabelList();
        property = (Expression) property.expandPropRefsAndLabels(propertiesFile, combinedLabelList);
        // Then get rid of any constants and simplify
        property = (Expression) property.replaceConstants(modulesFile.getConstantValues());
        property = (Expression) property.replaceConstants(propertiesFile.getConstantValues());
        property = (Expression) property.simplify();
        expressionList.add(property);
      }
    }
    constants.iterateProperty();

    return PrismParseResultTuple.create(modulesFile, propertiesFile, constants, expressionList);
  }

  public static explicit.Distribution scale(MapDistribution distribution) {
    double total = distribution.sum();
    if (Util.isEqual(total, 0.0d)) {
      return null;
    }
    if (Util.isEqual(total, 1.0d)) {
      return new explicit.Distribution(distribution.objectIterator());
    }
    Map<Integer, Double> map = new HashMap<>(distribution.size());
    for (Map.Entry<Integer, Double> entry : distribution) {
      map.put(entry.getKey(), entry.getValue() / total);
    }
    return new explicit.Distribution(map.entrySet().iterator());
  }

  @Value.Immutable
  @Tuple
  public abstract static class PrismParseResult {
    public abstract ModulesFile modulesFile();

    @Nullable
    public abstract PropertiesFile propertiesFile();

    public abstract UndefinedConstants constants();

    public abstract List<Expression> expressions();
  }
}
