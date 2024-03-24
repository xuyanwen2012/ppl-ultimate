target("cpu-only")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("main.cpp")
    add_packages("glm")
    add_deps("ppl")
    if is_plat("linux") then
        add_packages("pthread")
    end