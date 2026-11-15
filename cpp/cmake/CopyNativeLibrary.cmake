# volengine_copy_native_library(<target>): copies the native shared library next to an executable after it links.
#
# On Windows a DLL that is neither beside the executable nor on PATH is not found, and the process exits with
# 0xc0000135 before main runs. Everything this package builds calls this, and consumers should too. A no-op on Linux
# and macOS, where the rpath of the link resolves the library.
function(volengine_copy_native_library target)
    if(NOT WIN32)
        return()
    endif()
    if(NOT TARGET ${target})
        message(FATAL_ERROR "volengine_copy_native_library: '${target}' is not a target")
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:volengine::native>" "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying the VolEngine native library next to ${target}")
endfunction()
