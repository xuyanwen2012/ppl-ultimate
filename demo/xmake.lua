target("mini-demo")
    set_kind("binary")
--    add_includedirs("$(projectdir)/include")
    add_files("mini-demo/*.cpp")
    add_packages("glm", "pthread")