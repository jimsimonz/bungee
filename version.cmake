execute_process(
    COMMAND git -C ${GIT_CWD} describe --tags --always --abbrev=8 --dirty
    OUTPUT_VARIABLE GIT_DESCRIBE
    ERROR_QUIET)

if ("${GIT_DESCRIBE}" STREQUAL "")
    set(GIT_DESCRIBE "unknown")
else()
    string(STRIP "${GIT_DESCRIBE}" GIT_DESCRIBE)
endif()

set(CONTENTS "namespace ${VERSION_NAMESPACE} { const char *versionDescription = \"${GIT_DESCRIBE}\"; }")

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/version-backup/${VERSION_CPP})
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/version-backup/${VERSION_CPP} EXISTING_CONTENTS)
else()
    set(EXISTING_CONTENTS "")
endif()

if ("${CONTENTS}" STREQUAL "${EXISTING_CONTENTS}")
    #message("Restoring ${VERSION_NAMESPACE} ${VERSION_CPP} backup because it is still current with \"${GIT_DESCRIBE}\"")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/version-backup/${VERSION_CPP} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
else()
    #message("Generating ${VERSION_NAMESPACE} ${VERSION_CPP} with \"${GIT_DESCRIBE}\"")
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/${VERSION_CPP} "${CONTENTS}")
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/version-backup/${VERSION_CPP} "${CONTENTS}")
endif()
