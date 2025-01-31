#include "Participant.hpp"
#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "MappingContext.hpp"
#include "MeshContext.hpp"
#include "WatchIntegral.hpp"
#include "WatchPoint.hpp"
#include "action/Action.hpp"
#include "io/Export.hpp"
#include "logging/LogMacros.hpp"
#include "mesh/Data.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/config/DataConfiguration.hpp"
#include "mesh/config/MeshConfiguration.hpp"
#include "precice/impl/SharedPointer.hpp"
#include "precice/types.hpp"
#include "utils/ManageUniqueIDs.hpp"
#include "utils/String.hpp"
#include "utils/assertion.hpp"
#include "utils/fmt.hpp"

namespace precice::impl {

Participant::Participant(
    std::string                 name,
    mesh::PtrMeshConfiguration &meshConfig)
    : _name(std::move(name))
{
}

Participant::~Participant()
{
  for (MeshContext *context : _usedMeshContexts) {
    delete context;
  }
  _usedMeshContexts.clear();
}

/// Configuration interface

void Participant::addAction(action::PtrAction &&action)
{
  auto &context = meshContext(action->getMesh()->getName());
  context.require(action->getMeshRequirement());
  _actions.push_back(std::move(action));
}

void Participant::setUsePrimaryRank(bool useIntraComm)
{
  _useIntraComm = useIntraComm;
}

void Participant::addWatchPoint(
    const PtrWatchPoint &watchPoint)
{
  _watchPoints.push_back(watchPoint);
}

void Participant::addWatchIntegral(
    const PtrWatchIntegral &watchIntegral)
{
  _watchIntegrals.push_back(watchIntegral);
}

void Participant::provideMesh(const mesh::PtrMesh &mesh)
{
  std::string meshName = mesh->getName();
  PRECICE_TRACE(_name, meshName);
  checkDuplicatedUse(meshName);

  auto context                       = new MeshContext();
  context->mesh                      = mesh;
  context->provideMesh               = true;
  _meshContexts[std::move(meshName)] = context;
  _usedMeshContexts.push_back(context);
}

void Participant::receiveMesh(const mesh::PtrMesh &                         mesh,
                              const std::string &                           fromParticipant,
                              double                                        safetyFactor,
                              partition::ReceivedPartition::GeometricFilter geoFilter,
                              const bool                                    allowDirectAccess)
{
  std::string meshName = mesh->getName();
  PRECICE_TRACE(_name, meshName);
  checkDuplicatedUse(meshName);
  PRECICE_ASSERT(!fromParticipant.empty());
  PRECICE_ASSERT(safetyFactor >= 0);
  auto context               = new MeshContext();
  context->mesh              = mesh;
  context->receiveMeshFrom   = fromParticipant;
  context->safetyFactor      = safetyFactor;
  context->provideMesh       = false;
  context->geoFilter         = geoFilter;
  context->allowDirectAccess = allowDirectAccess;

  _meshContexts[std::move(meshName)] = context;

  _usedMeshContexts.push_back(context);
}

void Participant::addWriteData(
    const mesh::PtrData &data,
    const mesh::PtrMesh &mesh)
{
  checkDuplicatedData(mesh->getName(), data->getName());
  _writeDataContexts.emplace(MeshDataKey{mesh->getName(), data->getName()}, WriteDataContext(data, mesh));
}

void Participant::addReadData(
    const mesh::PtrData &data,
    const mesh::PtrMesh &mesh,
    int                  interpolationOrder)
{
  checkDuplicatedData(mesh->getName(), data->getName());
  _readDataContexts.emplace(MeshDataKey{mesh->getName(), data->getName()}, ReadDataContext(data, mesh, interpolationOrder));
}

void Participant::addReadMappingContext(
    const MappingContext &mappingContext)
{
  _readMappingContexts.push_back(mappingContext);
}

void Participant::addWriteMappingContext(
    const MappingContext &mappingContext)
{
  _writeMappingContexts.push_back(mappingContext);
}

// Data queries
const ReadDataContext &Participant::readDataContext(std::string_view mesh, std::string_view data) const
{
  auto it = _readDataContexts.find(MeshDataKey{mesh, data});
  PRECICE_CHECK(it != _readDataContexts.end(), "Data \"{}\" does not exist for mesh \"{}\".", data, mesh)
  return it->second;
}

ReadDataContext &Participant::readDataContext(std::string_view mesh, std::string_view data)
{
  auto it = _readDataContexts.find(MeshDataKey{mesh, data});
  PRECICE_CHECK(it != _readDataContexts.end(), "Data \"{}\" does not exist for mesh \"{}\".", data, mesh)
  return it->second;
}

const WriteDataContext &Participant::writeDataContext(std::string_view mesh, std::string_view data) const
{
  auto it = _writeDataContexts.find(MeshDataKey{mesh, data});
  PRECICE_CHECK(it != _writeDataContexts.end(), "Data \"{}\" does not exist in write direction.", data)
  return it->second;
}

WriteDataContext &Participant::writeDataContext(std::string_view mesh, std::string_view data)
{
  auto it = _writeDataContexts.find(MeshDataKey{mesh, data});
  PRECICE_CHECK(it != _writeDataContexts.end(), "Data \"{}\" does not exist in write direction.", data)
  return it->second;
}

bool Participant::hasData(std::string_view mesh, std::string_view data) const
{
  return std::any_of(
      _meshContexts.begin(), _meshContexts.end(),
      [data](const auto &mckv) {
        const auto &meshData = mckv.second->mesh->data();
        return std::any_of(meshData.begin(), meshData.end(), [data](const auto &dptr) {
          return dptr->getName() == data;
        });
      });
}

bool Participant::isDataUsed(std::string_view mesh, std::string_view data) const
{
  const auto &meshData = meshContext(mesh).mesh->data();
  const auto  match    = std::find_if(meshData.begin(), meshData.end(), [data](auto &dptr) { return dptr->getName() == data; });
  return match != meshData.end();
}

bool Participant::isDataRead(std::string_view mesh, std::string_view data) const
{
  return _readDataContexts.count(MeshDataKey{mesh, data}) > 0;
}

bool Participant::isDataWrite(std::string_view mesh, std::string_view data) const
{
  return _writeDataContexts.count(MeshDataKey{mesh, data}) > 0;
}

/// Mesh queries

const MeshContext &Participant::meshContext(std::string_view mesh) const
{
  auto pos = _meshContexts.find(mesh);
  PRECICE_ASSERT(pos != _meshContexts.end());
  return *pos->second;
}

MeshContext &Participant::meshContext(std::string_view mesh)
{
  auto pos = _meshContexts.find(mesh);
  PRECICE_ASSERT(pos != _meshContexts.end());
  return *pos->second;
}

const std::vector<MeshContext *> &Participant::usedMeshContexts() const
{
  return _usedMeshContexts;
}

std::vector<MeshContext *> &Participant::usedMeshContexts()
{
  return _usedMeshContexts;
}

MeshContext &Participant::usedMeshContext(std::string_view mesh)
{
  auto pos = std::find_if(_usedMeshContexts.begin(), _usedMeshContexts.end(),
                          [mesh](MeshContext const *context) {
                            return context->mesh->getName() == mesh;
                          });
  PRECICE_ASSERT(pos != _usedMeshContexts.end());
  return **pos;
}

MeshContext const &Participant::usedMeshContext(std::string_view mesh) const
{
  auto pos = std::find_if(_usedMeshContexts.begin(), _usedMeshContexts.end(),
                          [mesh](MeshContext const *context) {
                            return context->mesh->getName() == mesh;
                          });
  PRECICE_ASSERT(pos != _usedMeshContexts.end());
  return **pos;
}

bool Participant::hasMesh(std::string_view mesh) const
{
  return _meshContexts.count(mesh) > 0;
}

bool Participant::isMeshUsed(std::string_view mesh) const
{
  return std::any_of(
      _usedMeshContexts.begin(), _usedMeshContexts.end(),
      [mesh](const MeshContext *mcptr) {
        return mcptr->mesh->getName() == mesh;
      });
}

bool Participant::isMeshProvided(std::string_view mesh) const
{
  PRECICE_ASSERT(hasMesh(mesh));
  return usedMeshContext(mesh).provideMesh;
}

bool Participant::isMeshReceived(std::string_view mesh) const
{
  PRECICE_ASSERT(hasMesh(mesh));
  return !usedMeshContext(mesh).provideMesh;
}

bool Participant::isDirectAccessAllowed(std::string_view mesh) const
{
  PRECICE_ASSERT(hasMesh(mesh));
  return meshContext(mesh).allowDirectAccess;
}

// Other queries

std::vector<MappingContext> &Participant::readMappingContexts()
{
  return _readMappingContexts;
}

std::vector<MappingContext> &Participant::writeMappingContexts()
{
  return _writeMappingContexts;
}

std::vector<action::PtrAction> &Participant::actions()
{
  return _actions;
}

const std::vector<action::PtrAction> &Participant::actions() const
{
  return _actions;
}

void Participant::addExportContext(
    const io::ExportContext &exportContext)
{
  _exportContexts.push_back(exportContext);
}

const std::vector<io::ExportContext> &Participant::exportContexts() const
{
  return _exportContexts;
}

std::vector<PtrWatchPoint> &Participant::watchPoints()
{
  return _watchPoints;
}

std::vector<PtrWatchIntegral> &Participant::watchIntegrals()
{
  return _watchIntegrals;
}

bool Participant::useIntraComm() const
{
  return _useIntraComm;
}

const std::string &Participant::getName() const
{
  return _name;
}

void Participant::exportInitial()
{
  for (const io::ExportContext &context : exportContexts()) {
    if (context.everyNTimeWindows < 1) {
      continue;
    }

    for (const MeshContext *meshContext : usedMeshContexts()) {
      auto &mesh = *meshContext->mesh;
      PRECICE_DEBUG("Exporting initial mesh {} to location \"{}\"", mesh.getName(), context.location);
      context.exporter->doExport(fmt::format("{}-{}.init", mesh.getName(), getName()), context.location, mesh);
    }
  }
}

void Participant::exportFinal()
{
  for (const io::ExportContext &context : exportContexts()) {
    if (context.everyNTimeWindows < 1) {
      continue;
    }

    for (const MeshContext *meshContext : usedMeshContexts()) {
      auto &mesh = *meshContext->mesh;
      PRECICE_DEBUG("Exporting final mesh {} to location \"{}\"", mesh.getName(), context.location);
      context.exporter->doExport(fmt::format("{}-{}.final", mesh.getName(), getName()), context.location, mesh);
    }
  }
}

void Participant::exportIntermediate(IntermediateExport exp)
{
  for (const io::ExportContext &context : exportContexts()) {
    if (exp.complete && (context.everyNTimeWindows > 0) && (exp.timewindow % context.everyNTimeWindows == 0)) {
      for (const MeshContext *meshContext : usedMeshContexts()) {
        auto &mesh = *meshContext->mesh;
        PRECICE_DEBUG("Exporting mesh {} for timewindow {} to location \"{}\"", mesh.getName(), exp.timewindow, context.location);
        context.exporter->doExport(fmt::format("{}-{}.dt{}", mesh.getName(), getName(), exp.timewindow), context.location, mesh);
      }
    }

    if (context.everyIteration) {
      for (const MeshContext *meshContext : usedMeshContexts()) {
        auto &mesh = *meshContext->mesh;
        PRECICE_DEBUG("Exporting mesh {} for iteration {} to location \"{}\"", meshContext->mesh->getName(), exp.iteration, context.location);
        /// @todo this is the global iteration count. Shouldn't this be local to the timestep? example .dtN.itM or similar
        context.exporter->doExport(fmt::format("{}-{}.it{}", mesh.getName(), getName(), exp.iteration), context.location, mesh);
      }
    }
  }

  if (exp.complete) {
    // Export watch point data
    for (const PtrWatchPoint &watchPoint : watchPoints()) {
      watchPoint->exportPointData(exp.time);
    }

    for (const PtrWatchIntegral &watchIntegral : watchIntegrals()) {
      watchIntegral->exportIntegralData(exp.time);
    }
  }
}

// private

void Participant::checkDuplicatedUse(std::string_view mesh)
{
  PRECICE_CHECK(_meshContexts.count(mesh) == 0,
                "Mesh \"{} cannot be used twice by participant {}. "
                "Please remove one of the provide/receive-mesh nodes with name=\"{}\"./>",
                mesh, _name, mesh);
}

void Participant::checkDuplicatedData(std::string_view mesh, std::string_view data)
{
  PRECICE_CHECK(!isDataWrite(mesh, data) && !isDataRead(mesh, data),
                "Participant \"{}\" can read/write data \"{}\" from/to mesh \"{}\" only once. "
                "Please remove any duplicate instances of write-data/read-data nodes.",
                _name, mesh, data);
}

std::string Participant::hintForMesh(std::string_view mesh) const
{
  PRECICE_ASSERT(!hasMesh(mesh));
  PRECICE_ASSERT(!_meshContexts.empty());

  if (_meshContexts.size() == 1) {
    return " This participant only knows mesh \"" + _meshContexts.begin()->first + "\".";
  }

  auto matches = utils::computeMatches(mesh, _meshContexts | boost::adaptors::map_keys);
  if (matches.front().distance < 3) {
    return " Did you mean mesh \"" + matches.front().name + "\"?";
  } else {
    return fmt::format(" Available meshes are: {}", fmt::join(_meshContexts | boost::adaptors::map_keys, ", "));
  }
}

std::string Participant::hintForMeshData(std::string_view mesh, std::string_view data) const
{
  PRECICE_ASSERT(hasMesh(mesh));
  PRECICE_ASSERT(!hasData(mesh, data));
  PRECICE_ASSERT(!_meshContexts.empty());

  // Is there such data in other meshes?
  std::vector<std::string> otherMeshes;
  for (const auto &[_, mc] : _meshContexts) {
    if (mc->mesh->hasDataName(data)) {
      return " Did you mean the data of mesh \"" + mc->mesh->getName() + "\"?";
    }
  }

  // Is there other data in the given mesh?
  auto localData = meshContext(mesh).mesh->availableData();

  if (localData.size() == 1) {
    return " This mesh only knows data \"" + localData.front() + "\".";
  }

  // Was the data typoed?
  auto matches = utils::computeMatches(mesh, localData);
  if (matches.front().distance < 3) {
    return " Did you mean data \"" + matches.front().name + "\"?";
  }

  return fmt::format(" Available data are: {}", fmt::join(localData, ", "));
}

} // namespace precice::impl
