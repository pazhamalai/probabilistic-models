Usage in other projects - compiling PRISM
=========================================

When using this project in another project perform the following steps.
(If not already happened) Init the PRISM repo `git submodule update --init --recursive`.
Then, run `make` in `<path>/lib/prism/prism` to compile PRISM.
For windows users:
  1. Install cygwin
  2. Run `cygwin.bat` in the install dir to start cygwin bash
  3. `cd /cygdrive/path/to/lib/prism/prism`
  4. `dos2unix * ../cudd/*` (to fix incorrect line endings if checked out via windows git and not cygwin git) 
  5. `make JAVA_DIR=/cygdrive/path/to/jdk JAVAC=/cygdrive/path/to/jdk/bin/javac`
    - Make sure that this JDK is <= the version you are using to develop.
    - Put the paths in quotes.
    - If there are spaces in the JAVAC path, escape them with `\` (on top of quotes!)

Setup for other projects
=======================

1. Add this repository as a submodule `git submodule add <url> <path>`, `<path>` could for example be `lib/models`
2. Init the submodule (and the referenced PRISM repo) `git submodule update --init --recursive`
3. Compile PRISM as described above.
4. Add this project in your `settings.gradle` with `include '<path>'` (use `:` instead of `/`, for example `lib:models`)
5. Add it as dependency in `build.gradle` with `implementation project('<path>')`
6. Run `./gradlew compileJava` to check that everything is working.

Developer Instructions
======================

1. Clone the repository
2. Run `git submodule update --init` to populate `lib/prism`
3. Compile PRISM as described above.
4. Import project in IntelliJ, enable gradle auto-import
5. Run `./gradlew compileJava` to create the prism.jar and run the annotation processor
