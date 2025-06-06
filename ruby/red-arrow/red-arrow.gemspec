# -*- ruby -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require_relative "lib/arrow/version"

Gem::Specification.new do |spec|
  is_jruby = RUBY_ENGINE == "jruby"

  spec.name = "red-arrow"
  spec.platform = "java" if is_jruby

  version_components = [
    Arrow::Version::MAJOR.to_s,
    Arrow::Version::MINOR.to_s,
    Arrow::Version::MICRO.to_s,
    Arrow::Version::TAG,
  ]
  spec.version = version_components.compact.join(".")
  spec.homepage = "https://arrow.apache.org/"
  spec.authors = ["Apache Arrow Developers"]
  spec.email = ["dev@arrow.apache.org"]

  spec.summary = "Red Arrow is the Ruby bindings of Apache Arrow"
  spec.description =
    "Apache Arrow is a common in-memory columnar data store. " +
    "It's useful to share and process large data."
  spec.license = "Apache-2.0"
  spec.files = ["README.md", "Rakefile", "Gemfile", "#{spec.name}.gemspec"]
  spec.files += ["LICENSE.txt", "NOTICE.txt"]
  spec.files += Dir.glob("ext/**/*.{cpp,hpp,rb}")
  spec.files += Dir.glob("lib/**/*.rb")
  spec.files += Dir.glob("image/*.*")
  spec.files += Dir.glob("doc/text/*")
  spec.extensions = ["ext/arrow/extconf.rb"] unless is_jruby

  required_arrow_glib_version = version_components[0, 3].join(".")

  spec.add_runtime_dependency("bigdecimal", ">= 3.1.0")
  spec.add_runtime_dependency("csv")
  if is_jruby
    spec.add_runtime_dependency("jar-dependencies")
    spec.requirements << "jar org.apache.arrow, arrow-vector, #{spec.version}"
    spec.requirements << "jar org.apache.arrow, arrow-memory-netty, #{spec.version}"
  else
    spec.add_runtime_dependency("extpp", ">= 0.1.1")
    spec.add_runtime_dependency("gio2", ">= 4.2.3")
    spec.add_runtime_dependency("pkg-config")

    repository_url_prefix = "https://packages.apache.org/artifactory/arrow"
    [
      # Try without additional repository
      ["amazon_linux", "arrow-glib-devel"],
      # Retry with additional repository
      [
        "amazon_linux",
        "#{repository_url_prefix}/amazon-linux/%{version}/" +
        "apache-arrow-release-latest.rpm",
      ],
      ["amazon_linux", "arrow-glib-devel"],

      # Try without additional repository
      ["centos", "arrow-glib-devel"],
      # Retry with additional repository
      [
        "centos",
        "#{repository_url_prefix}/centos/%{major_version}-stream/" +
        "apache-arrow-release-latest.rpm",
      ],
      ["centos", "arrow-glib-devel"],

      ["conda", "arrow-c-glib"],

      # Try without additional repository
      ["debian", "libarrow-glib-dev"],
      # Retry with additional repository
      [
        "debian",
        "#{repository_url_prefix}/%{distribution}/" +
        "apache-arrow-apt-source-latest-%{code_name}.deb",
      ],
      ["debian", "libarrow-glib-dev"],

      ["fedora", "libarrow-glib-devel"],

      # Try without additional repository
      ["rhel", "arrow-glib-devel"],
      # Retry with additional repository
      [
        "rhel",
        "#{repository_url_prefix}/almalinux/%{major_version}/" +
        "apache-arrow-release-latest.rpm",
      ],
      ["rhel", "arrow-glib-devel"],
    ].each do |platform, package|
      spec.requirements <<
        "system: arrow-glib>=#{required_arrow_glib_version}: " +
        "#{platform}: #{package}"
    end
  end

  spec.metadata["msys2_mingw_dependencies"] =
    "arrow>=#{required_arrow_glib_version}"
end
