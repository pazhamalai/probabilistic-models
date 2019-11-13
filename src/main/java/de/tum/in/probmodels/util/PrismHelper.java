package de.tum.in.probmodels.util;

import de.tum.in.probmodels.util.annotation.Tuple;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
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

  public static PrismParseResult parse(CommandLine commandLine, Option modelOption,
      @Nullable Option propertiesOption, Option constantsOption)
      throws IOException, PrismException {
    String modelPath = commandLine.getOptionValue(modelOption.getLongOpt());
    @Nullable
    String propertiesPath = propertiesOption == null
        ? null : commandLine.getOptionValue(propertiesOption.getLongOpt());
    @Nullable
    String constantsString = commandLine.hasOption(constantsOption.getLongOpt())
        ? commandLine.getOptionValue(constantsOption.getLongOpt())
        : null;
    return parse(modelPath, propertiesPath, constantsString);
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
