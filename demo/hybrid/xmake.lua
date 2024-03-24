target("hybrid")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("demo.cpp", "main.cu")
    add_packages("glm", "pthread")
    add_deps("ppl-hybrid")
    add_cugencodes("native")
