<template>
    <div>
        <v-expansion-panel @input="(expanded) => expandPanel(0, expanded)" :value="[ui.expandedPanels.includes(0)]" expand>
            <v-expansion-panel-content>
                <div slot="header" class="title">Common Properties</div>
                <v-card class="px-4 pt-1 pb-4">
                    <v-checkbox
                        v-model="showCandidateKeyphrases"
                        label="candidate keyphrases"
                        hide-details
                        class="mt-1"
                    />
                    <v-checkbox
                        v-model="showSelectedKeyphrases"
                        label="selected keyphrases"
                        hide-details
                        class="mt-1"
                    />
                </v-card>
            </v-expansion-panel-content>
        </v-expansion-panel>
        <v-expansion-panel v-if="selectedRun.settings.model === 'KeyCluster'" @input="(expanded) => expandPanel(1, expanded)" :value="[ui.expandedPanels.includes(1)]" expand class="mt-3 settings-panel">
            <v-expansion-panel-content>
                <div slot="header" class="title">KeyCluster Properties</div>
                <v-card class="px-3 pb-3">
                    <PropertyList :items="clusters" :headers="headers" v-model="selectedClusters"/>
                </v-card>
            </v-expansion-panel-content>
        </v-expansion-panel>
        <v-expansion-panel @input="(expanded) => expandPanel(2, expanded)" :value="[ui.expandedPanels.includes(2)]" expand class="mt-3 settings-panel">
            <v-expansion-panel-content class="settings-panel">
                <div slot="header" class="title">Run Settings</div>
                <v-card class="px-3 pb-3">
                    <SettingsList :settings="this.selectedRun.settings" />
                </v-card>
            </v-expansion-panel-content>
        </v-expansion-panel>
        <v-expansion-panel @input="(expanded) => expandPanel(3, expanded)" :value="[ui.expandedPanels.includes(3)]" expand class="mt-3 settings-panel">
            <v-expansion-panel-content class="settings-panel">
                <div slot="header" class="title">Runs</div>
                <v-card class="px-3 pb-3">
                    <RunList />
                </v-card>
            </v-expansion-panel-content>
        </v-expansion-panel>
    </div>
</template>

<script>
import { mapActions, mapState, mapGetters } from 'vuex';
import PropertyList from './PropertyList.vue';
import SettingsList from './SettingsList.vue';
import RunList from './RunList.vue';

export default {
    components: {
        PropertyList,
        SettingsList,
        RunList,
    },
    computed: {
        ...mapState('document', [
            'document',
            'ui',
        ]),
        ...mapGetters('document', [
            'selectedRun',
        ]),
        showCandidateKeyphrases: {
            get() {
                return this.ui.commonProperties.showCandidateKeyphrases;
            },
            set(value) {
                this.setCommonProperty({ showCandidateKeyphrases: value });
            },
        },
        showSelectedKeyphrases: {
            get() {
                return this.ui.commonProperties.showSelectedKeyphrases;
            },
            set(value) {
                this.setCommonProperty({ showSelectedKeyphrases: value });
            },
        },
        selectedClusters: {
            get() {
                return this.ui.keyClusterProperties.selectedClusters.map(entry => ({
                    name: `cluster ${entry}`,
                    value: entry,
                }));
            },
            set(value) {
                const selectedIds = value.map(entry => entry.value);
                this.setKeyClusterProperty({ selectedClusters: selectedIds });
            },
        },
        clusters() {
            return [...Array(this.selectedRun.num_clusters + 1).keys()].slice(1).map(entry => ({
                name: `cluster ${entry}`,
                value: entry,
            }));
        },
    },
    data() {
        return {
            allSelectedState: false,
            allSelectedIndetermined: false,
            headers: [
                {
                    text: 'Cluster',
                    align: 'left',
                    sortable: false,
                },
            ],
        };
    },
    methods: {
        ...mapActions('document/ui', [
            'setCommonProperty',
            'setKeyClusterProperty',
            'setPanelExpanded',
        ]),
        expandPanel(panel, expanded) {
            this.setPanelExpanded({ panel, expanded: expanded[0] });
        },
    },
};
</script>

<style>
.fixed-header th:after {
    content:'';
    position:absolute;
    left: 0;
    bottom: 0;
    width:100%;
    border-bottom: 1px solid rgba(0,0,0,0.12);
}

.fixed-header thead tr:first-child {
    border: none !important;
}

.fixed-header th {
    background-color: #fff; /* just for LIGHT THEME, change it to #474747 for DARK */
    position: sticky;
    top: 0;
    z-index: 1;
}

.fixed-header tr.v-datatable__progress th {
    top: 56px;
    position: absolute;
}

.fixed-header .v-table__overflow {
    overflow: auto;
    height: 224px;
}
</style>
