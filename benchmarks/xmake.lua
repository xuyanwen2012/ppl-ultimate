

package("benchmark")
    set_kind("library")
    add_deps("cmake") 
    set_urls("https://github.com/google/benchmark.git")
    add_versions("v1.5.0", "06b4a070156a9333549468e67923a3a16c8f541b") 

    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=on")
        table.insert(configs, "-DHAVE_THREAD_SAFETY_ATTRIBUTES=0")
        import("package.tools.cmake").install(package, configs)
    end)
package_end()



add_requires("benchmark")

target("bench-cpu")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_headerfiles("*.hpp")
    add_files("*.cpp")
    add_packages("benchmark", "glm")
    add_deps("ppl")
    add_cugencodes("native")


-- target("bench-gpu")
--     set_kind("binary")
--     add_includedirs("$(projectdir)/include")
--     add_files("*.cu")
--     add_cugencodes("native")
--     add_packages("benchmark")
--     add_deps("ppl-hybrid")
    