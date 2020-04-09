Usage in other projects
=======================

When using this project in another project perform the following steps.
(If not already happened) Init the PRISM repo `git submodule update --init --recursive`.
Then, run `make` in `<path>/lib/prism/prism` to compile PRISM.
For windows users:
  1. Install cygwin
  2. Run `cygwin.bat` in the install dir to start cygwin bash
  3. `cd /cygdrive/path/to/lib/prism/prism`
  4. `dos2unix * ../cudd/*` (to potentially fix incorrect line endings)
  5. `make`. If that doesn't work: `make JAVA_DIR=/cygdrive/path/to/jdk JAVAC=/cygdrive/path/to/jdk/bin/javac`
    - Make sure that this JDK is <= the version you are using to develop.
    - Put the paths in quotes.
    - If there are spaces in the JAVAC path, escape them with `\` (on top of quotes!)
  6. If you experience problems with PRISM not finding its dlls, copy them into the CWD of the command.
 
Setup for other projects
=======================

1. Add this repository as a submodule `git submodule add <url> <path>`, `<path>` could for example be `lib/models`
2. Init the submodule (and the referenced PRISM repo) `git submodule update --init --recursive`
3. Compile PRISM as described above.
4. Add this project in your `settings.gradle` with `include '<path>'` (use `:` instead of `/`, for example `lib:models`)
5. Add it as dependency in `build.gradle` with `implementation project('<path>')`
6. Run `./gradlew compileJava` to check that everything is working.
7. (if using the `application` gradle plugin) Add the following code to your `build.gradle` to include the PRISM native libraries in the generated application
   ```groovy
    task extractScriptTemplates {
        doLast {
            file("<path>/config/template-unix.default.txt").text =
                    startScripts.unixStartScriptGenerator.template.asString()
            file("<path>/config/template-windows.default.txt").text =
                    startScripts.windowsStartScriptGenerator.template.asString()
        }
    }
    
    startScripts {
        unixStartScriptGenerator.template = resources.text.fromFile("<path>/config/template-unix.txt")
        windowsStartScriptGenerator.template = resources.text.fromFile("<path>/config/template-windows.txt")
    }
    
    distributions {
        main {
            contents {
                from("$rootProject.buildDir/lib/") {
                    include "*.dll"
                    include "*.so"
                    into "lib"
                }
            }
        }
    }

   ```
Developer Instructions
======================

1. Clone the repository
2. Run `git submodule update --init` to populate `lib/prism`
3. Compile PRISM as described above.
4. Import project in IntelliJ, enable gradle auto-import
5. Run `./gradlew compileJava` to create the prism.jar and run the annotation processor
