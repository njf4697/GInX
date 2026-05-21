using namespace RaytracingX;

template <typename StructType>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE CCTK_ATTRIBUTE_ALWAYS_INLINE int RaytracingParticlesContainer<StructType>::get_interpolation_center(const CCTK_REAL point, const CCTK_REAL lower, const CCTK_REAL upper, const CCTK_REAL dx)
{
    int i = amrex::Math::floor((amrex::Clamp(point, lower, upper - dx / 4) - lower) / dx);
    ASSERT_FINITE(i)
    return i;
}

template <typename StructType>
void RaytracingParticlesContainer<StructType>::write_to_one_file(std::string filename, std::string data)
{
    const int proc_id = amrex::ParallelDescriptor::MyProc();
    const int nprocs = amrex::ParallelDescriptor::NProcs();
    const int data_size = data.size();

    // check to make sure we actually have something to write
    int has_data = (data_size > 0);
    amrex::ParallelDescriptor::ReduceIntSum(has_data);
    if (!has_data)
    {
        return;
    }

    // get sizes of each string
    std::vector<int> recv_sizes;
    if (proc_id == amrex::ParallelDescriptor::IOProcessorNumber())
    {
        recv_sizes.resize(nprocs);
    }
    amrex::ParallelDescriptor::Gather(&data_size, 1, recv_sizes.data(), 1, amrex::ParallelDescriptor::IOProcessorNumber());

    // get displacements of each string and set size for recieving buffer
    std::vector<int> displacements;
    std::vector<char> recv_buffer;
    if (proc_id == amrex::ParallelDescriptor::IOProcessorNumber())
    {
        displacements.resize(nprocs);

        int total = 0;
        for (int i = 0; i < nprocs; ++i)
        {
            displacements[i] = total;
            total += recv_sizes[i];
        }

        recv_buffer.resize(total);
    }

    // gather strings from all procs
    MPI_Gatherv(data.data(), data_size, MPI_CHAR, recv_buffer.data(), recv_sizes.data(), displacements.data(), MPI_CHAR, amrex::ParallelDescriptor::IOProcessorNumber(), amrex::ParallelDescriptor::Communicator());

    // output
    if (proc_id == amrex::ParallelDescriptor::IOProcessorNumber())
    {
        std::ofstream file;
        file.open(filename, std::ios::app);

        if (!file.is_open())
        {
            CCTK_VERROR("Could not open file %s", filename.c_str());
            return;
        }

        file.write(recv_buffer.data(), recv_buffer.size());
    }
}

void RaytracingParticlesContainer<StructType>::write_deleted_particle_data(const int &lev, std::string final_data_file_name)
{
    std::string output_str;

    for (GInX::ParticleIterator<StructType> pti(*this, lev); pti.isValid();
         ++pti)
    {
        const int np = pti.numParticles();

        // Get the information relate to the velocities and energy.
        auto &attribs = pti.GetAttributes();
        CCTK_REAL *AMREX_RESTRICT vels_x = attribs[StructType::vx].data();
        CCTK_REAL *AMREX_RESTRICT vels_y = attribs[StructType::vy].data();
        CCTK_REAL *AMREX_RESTRICT vels_z = attribs[StructType::vz].data();
        CCTK_REAL *AMREX_RESTRICT ln_alphaenergy = attribs[StructType::ln_alphaE].data();
        CCTK_REAL *AMREX_RESTRICT tau = attribs[StructType::tau].data();                          // RaytracingX: Add optical depth.
        CCTK_REAL *AMREX_RESTRICT index = attribs[StructType::pixel_number].data();               // RaytracingX: Add pixel index.
        CCTK_REAL *AMREX_RESTRICT deletion_reasons = attribs[StructType::deletion_reason].data(); // RaytracingX: Add deletion reason.
        auto *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        for (int i = 0; i < np; i++)
        {
            if (particles[i].id() != -1)
                continue;

            output_str += std::to_string((int)index[i]) + "\t"
                        + std::to_string(particles[i].pos(0)) + "\t"
                        + std::to_string(particles[i].pos(1)) + "\t"
                        + std::to_string(particles[i].pos(2)) + "\t"
                        + std::to_string(vels_x[i]) + "\t"
                        + std::to_string(vels_y[i]) + "\t"
                        + std::to_string(vels_z[i]) + "\t"
                        + std::to_string(ln_alphaenergy[i]) + "\t"
                        + std::to_string(tau[i]) + "\t"
                        + std::to_string((int)deletion_reasons[i]) + "\n";
        }
    }

    write_to_one_file(final_data_file_name, output_str);
}