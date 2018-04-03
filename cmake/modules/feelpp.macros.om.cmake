#create the directory of the lib

set( FMU_FILE "${OMWRAPPER_LIBDIR}/${OMWRAPPER_NAME}.fmu" )

# generate the library using the custom makefile
message( "Generating FMU for OM model ${OMWRAPPER_NAME}" )
execute_process( COMMAND ${OMC_COMPILER} ${FMU_SCRIPT_NAME} WORKING_DIRECTORY ${OMWRAPPER_LIBDIR} OUTPUT_QUIET )
