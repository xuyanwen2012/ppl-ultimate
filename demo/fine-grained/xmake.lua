target("fine-grained")
    set_kind("binary")
    add_includedirs("$(projectdir)/include")
    add_files("*.cpp")
    add_packages("glm")
    if(is_plat("linux")) then
        add_packages("pthread")
    end
    add_deps("ppl")
